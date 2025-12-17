#!/usr/bin/env python
# coding: utf-8

"""
Created on Friday January 24 13:33:04 2025

@author: Pratik
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
#import matplotlib.pyplot as plt
from scipy.stats import norm
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utils.utilities3 import *

def rbf_kernel(x1, x2, lengthscale=0.01, variance=0.8):
    sqdist = (x1 - x2.T)**2
    return variance * torch.exp(-0.5 * sqdist / lengthscale**2)

n = 2048

# Generate positions (can be 1D spatial points)
x = torch.linspace(0, 1, n).unsqueeze(1)

# RBF Kernel function
std = 0.1 + 0.3 * torch.sin(2 * math.pi * x).squeeze()**2  # shape: (n,)
D = torch.diag(std)

# Construct covariance matrix (2048x2048)
cov = rbf_kernel(x, x)
Sigma = D @ cov @ D




ntest = 50
z = norm.ppf(0.95)
pred = np.load("pred/burger-FNO-lcl_dat-lcl.npy")
var = np.load("pred/burger-FNO-lcl_dat-lcl_cov.npy")
# print(var[0,:10,:10])
y_test = np.load("pred/burger_test_lcl.npy")
mse = np.mean((y_test - pred) ** 2)
print("MSE FNO Local:", mse)
print("MPIW:", np.mean(2*z*var))
# print(y_test.shape)
data = np.load("datasets/generated_1d_data_Burger-04_matern.npy")
y_data = data[-ntest:,:,9]
diff = pred - y_test

x = np.array(range(2**11))/2**11 - 1
index = [1,37,45]


sns.set(style="whitegrid")
for i in index:
    mean = pred[i, :]                  # shape (D,)
    std = var[i, :]                 # shape (D, D)
    
    lower = mean - z * std
    upper = mean + z * std
    x1 = torch.linspace(0, 1, n)
    time_init1 = np.array([1.0])

    df_y = pd.DataFrame({"val" : np.transpose(y_test[i,:]).reshape(y_test.shape[1]),
                   "x" : x})

    df_p = pd.DataFrame({"val" : np.transpose(pred[i,:]).reshape(y_test.shape[1]),
                       #"type" : np.repeat(["obs.","pred."],y_test.shape[1]*y_test.shape[2]),
                       "x" : x})
    df_bound = pd.DataFrame({"lb" : lower.reshape(y_test.shape[1]),
                                "ub" : upper.reshape(y_test.shape[1]),
                       "x" : x})
    ax = sns.lineplot(x= "x", y="val", data=df_y,linewidth=0.5, color = "grey", label = "True")
    
    ax = sns.lineplot(x= "x", y="val", data=df_p, linestyle = "dotted", color = "black", label = "Pred")
    
    plt.fill_between(df_bound["x"], df_bound["lb"], df_bound["ub"], color='gray', alpha=0.3, label="90% PI pred")
    
    ax.set_xlabel('$s$',fontsize=20)
    ax.set_ylabel('$Y_t^{(L)} \quad vs \quad \hat{Y}_t^{(L)}$',fontsize=20)

    # Add legend
    ax.legend(loc="upper left")
    ax.set_xticks([]) 
    plt.savefig("plots/burger-FNO-lcl_dat-lcl-"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    plt.clf()

    ax = sns.lineplot(x= x, y= diff[i,:], linewidth=2, color = "black")
    #ax.set_title("$Y_t^{(L)} \quad - \quad \hat{Y}_t^{(L)}$", fontsize=20)
    ax.set_xlabel('$s$',fontsize=20)
    ax.set_xticks([]) 
    plt.savefig("plots/diff_burger-FNO-lcl_dat-lcl-"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    plt.clf()
    
    
    
    
    
    
     
    
    
    
pred = np.load("pred/burger-FNO-lcl_dat-nonlcl.npy")
var = np.load("pred/burger-FNO-lcl_dat-nonlcl_cov.npy")
# print(var.shape)
y_test = np.load("pred/burger_test_nonlcl.npy")
mse = np.mean((y_test - pred) ** 2)
print("MSE FNO Local data nonlocal:", mse)
print("MPIW:", np.mean(2*z*var))
# print(y_test.shape)
data = np.load("datasets/generated_1d_data_Burger-beta~U<0.05-0.7>_matern.npy")
y_data = data[-ntest:,:,9]
diff = pred - y_test
#data = np.load("generated_1d_data_Burger-04_matern.npy")
#x_test = data[950:,:,0]
x = np.array(range(2**11))/2**11 - 1
# index = [0,25,45]
plt.figure(figsize=(10, 6))

sns.set(style="whitegrid")
for i in index:
             # shape: [D]
    mean = pred[i, :]                  # shape (D,)
    # cov = var[i, :]                 # shape (D, D)
    # print(cov.shape)
    # sns.heatmap(cov, cmap="viridis", square=True, cbar_kws={'label': 'Covariance'})
    # plt.imshow(cov, cmap='viridis', aspect='auto')
    # plt.colorbar(label="Covariance")
    # plt.title("Covariance Matrix (imshow)")
    # plt.xlabel("Index")
    # plt.ylabel("Index")
    # plt.savefig("plots/burger-FNO-lcl_dat-nonlcl-matrix"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    # plt.clf()
    std =  var[i, :]
    # print(std)
    lower = mean - z * std
    upper = mean + z * std
    x1 = torch.linspace(0, 1, n)

    # RBF Kernel function
    std = 0.08 * torch.ones_like(x1)
    lower_t = y_data[i,:] - z * std.numpy()
    upper_t = y_data[i,:] + z * std.numpy()
    time_init1 = np.array([1.0])
    # print(y_test[i,:].shape)
    df_y = pd.DataFrame({"val" : np.transpose(y_test[i,:]),
                   "x" : x})

    df_p = pd.DataFrame({"val" : np.transpose(pred[i,:]).reshape(y_test.shape[1]),
                       #"type" : np.repeat(["obs.","pred."],y_test.shape[1]*y_test.shape[2]),
                       "x" : x})
    df_bound = pd.DataFrame({"lb" : lower.reshape(y_test.shape[1]),
                                "ub" : upper.reshape(y_test.shape[1]),
                       "x" : x})
    df_bound_t = pd.DataFrame({"lb" : lower_t.reshape(y_test.shape[1]),
                                "ub" : upper_t.reshape(y_test.shape[1]),
                       "x" : x})
    # ax = sns.lineplot(x= "x", y="val", data=df, hue = "time",linewidth=2,palette="flare")
    ax = sns.lineplot(x= "x", y="val", data=df_y,linewidth=2, color = "grey", label = "True")
    # ax = sns.lineplot(x= "x", y="val", data=df_p, hue = "time",linestyle = "dotted",linewidth=2,palette="crest_r")
    ax = sns.lineplot(x= "x", y="val", data=df_p, linestyle = "dotted", color = "black", label = "Pred")
    # ax = sns.lineplot(x= x, y=var[i,:], linestyle = "dotted", color = "red", label = "Var")
    #ax = sns.lineplot(x= "x", y="out", data=df, linestyle = "dotted", hue = "time",linewidth=2)
    plt.fill_between(df_bound["x"], df_bound["lb"], df_bound["ub"], color='gray', alpha=0.3, label="90% PI")
    plt.fill_between(df_bound_t["x"], df_bound_t["lb"], df_bound_t["ub"], color='steelblue', alpha=0.3, label="90% PI true")
    ax.set_xlabel('$s$',fontsize=20)
    ax.set_ylabel('$Y_t^{(L)} \quad vs \quad \hat{Y}_t^{(L)}$',fontsize=20)
    # Add title
    # ax.set_title("Instance "+str(i), fontsize=20)

    # Add legend
    ax.legend(loc="upper left")
    ax.set_xticks([]) 
   
    plt.savefig("plots/burger-FNO-lcl_dat-nonlcl-"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    plt.clf()

    ax = sns.lineplot(x= x, y= diff[i,:], linewidth=2, color = "black")
    #ax.set_title("$Y_t^{(L)} \quad - \quad \hat{Y}_t^{(L)}$", fontsize=20)
    ax.set_xlabel('$s$',fontsize=20)
    ax.set_xticks([]) 
    plt.savefig("plots/diff_burger-FNO-lcl_dat-nonlcl-"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    plt.clf()






pred = np.load("pred/burger-FNO-nonlcl_dat-nonlcl.npy")
var = np.load("pred/burger-FNO-nonlcl_dat-nonlcl_cov.npy")
# print(var.shape)
y_test = np.load("pred/burger_test_nonlcl-FNO-nonlcl.npy")
mse = np.mean((y_test - pred) ** 2)
print("MSE FNO nonlocal:", mse)
print("MPIW:", np.mean(2*z*var))
# print(y_test.shape)
data = np.load("datasets/generated_1d_data_Burger-beta~U<0.05-0.7>_matern.npy")
y_data = data[-ntest:,:,9]
diff = pred - y_test
#data = np.load("generated_1d_data_Burger-04_matern.npy")
#x_test = data[950:,:,0]
x = np.array(range(2**11))/2**11 - 1
# index = [0,25,45]
plt.figure(figsize=(10, 6))

sns.set(style="whitegrid")
for i in index:
             # shape: [D]
    mean = pred[i, :]                  # shape (D,)
    # cov = var[i, :]                 # shape (D, D)
    # print(cov.shape)
    # sns.heatmap(cov, cmap="viridis", square=True, cbar_kws={'label': 'Covariance'})
    # plt.imshow(cov, cmap='viridis', aspect='auto')
    # plt.colorbar(label="Covariance")
    # plt.title("Covariance Matrix (imshow)")
    # plt.xlabel("Index")
    # plt.ylabel("Index")
    # plt.savefig("plots/burger-FNO-lcl_dat-nonlcl-matrix"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    # plt.clf()
    std =  var[i, :]
    # print(std)
    lower = mean - z * std
    upper = mean + z * std
    x1 = torch.linspace(0, 1, n)

    # RBF Kernel function
    std = 0.08 * torch.ones_like(x1)
    # std = 0.25 * x1 + 1e-2 * torch.ones_like(x1)
    lower_t = y_data[i,:] - z * std.numpy()
    upper_t = y_data[i,:] + z * std.numpy()
    time_init1 = np.array([1.0])
    # print(y_test[i,:].shape)
    df_y = pd.DataFrame({"val" : np.transpose(y_test[i,:]),
                   "x" : x})
    df_p = pd.DataFrame({"val" : np.transpose(pred[i,:]).reshape(y_test.shape[1]),
                       #"type" : np.repeat(["obs.","pred."],y_test.shape[1]*y_test.shape[2]),
                       "x" : x})

    df_bound = pd.DataFrame({"lb" : lower.reshape(y_test.shape[1]),
                                "ub" : upper.reshape(y_test.shape[1]),
                       "x" : x})
    df_bound_t = pd.DataFrame({"lb" : lower_t.reshape(y_test.shape[1]),
                                "ub" : upper_t.reshape(y_test.shape[1]),
                       "x" : x})
    # ax = sns.lineplot(x= "x", y="val", data=df, hue = "time",linewidth=2,palette="flare")
    ax = sns.lineplot(x= "x", y="val", data=df_y,linewidth=2, color = "grey", label = "True")
    # ax = sns.lineplot(x= "x", y="val", data=df_p, hue = "time",linestyle = "dotted",linewidth=2,palette="crest_r")
    ax = sns.lineplot(x= "x", y="val", data=df_p, linestyle = "dotted", color = "black", label = "Pred")
    #ax = sns.lineplot(x= "x", y="out", data=df, linestyle = "dotted", hue = "time",linewidth=2)
    plt.fill_between(df_bound["x"], df_bound["lb"], df_bound["ub"], color='gray', alpha=0.3, label="90% PI")
    plt.fill_between(df_bound_t["x"], df_bound_t["lb"], df_bound_t["ub"], color='steelblue', alpha=0.3, label="90% PI true")
    ax.set_xlabel('$s$',fontsize=20)
    ax.set_ylabel('$Y_t^{(L)} \quad vs \quad \hat{Y}_t^{(L)}$',fontsize=20)
    # Add title
    # ax.set_title("Instance "+str(i), fontsize=20)

    # Add legend
    ax.legend(loc="upper left")
    ax.set_xticks([]) 
   
    plt.savefig("plots/burger-FNO-nonlcl_dat-nonlcl-"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    plt.clf()

    ax = sns.lineplot(x= x, y= diff[i,:], linewidth=2, color = "black")
    #ax.set_title("$Y_t^{(L)} \quad - \quad \hat{Y}_t^{(L)}$", fontsize=20)
    ax.set_xlabel('$s$',fontsize=20)
    ax.set_xticks([]) 
    plt.savefig("plots/diff_burger-FNO-nonlcl_dat-nonlcl-"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    plt.clf()



pred = np.load("pred/burger-FNO-nonlcl_dat-lcl.npy")
var = np.load("pred/burger-FNO-nonlcl_dat-lcl_cov.npy")
# print(var.shape)
y_test = np.load("pred/burger_test_nonlcl-FNO-lcl.npy")
mse = np.mean((y_test - pred) ** 2)
print("MSE FNO nonlocal data local:", mse)
print("MPIW:", np.mean(2*z*var))
# print(y_test.shape)
data = np.load("datasets/generated_1d_data_Burger-04_matern.npy")
y_data = data[-ntest:,:,9]
diff = pred - y_test
#data = np.load("generated_1d_data_Burger-04_matern.npy")
#x_test = data[950:,:,0]
x = np.array(range(2**11))/2**11 - 1
# index = [0,25,45]
plt.figure(figsize=(10, 6))

sns.set(style="whitegrid")
for i in index:
             # shape: [D]
    mean = pred[i, :]                  # shape (D,)
    # cov = var[i, :]                 # shape (D, D)
    # print(cov.shape)
    # sns.heatmap(cov, cmap="viridis", square=True, cbar_kws={'label': 'Covariance'})
    # plt.imshow(cov, cmap='viridis', aspect='auto')
    # plt.colorbar(label="Covariance")
    # plt.title("Covariance Matrix (imshow)")
    # plt.xlabel("Index")
    # plt.ylabel("Index")
    # plt.savefig("plots/burger-FNO-lcl_dat-nonlcl-matrix"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    # plt.clf()
    std =  var[i, :]
    # print(std)
    lower = mean - z * std
    upper = mean + z * std
    x1 = torch.linspace(0, 1, n)

    # RBF Kernel function
    std = 0.08 * torch.ones_like(x1)
    # std = 0.25 * x1 + 1e-2 * torch.ones_like(x1)
    lower_t = y_data[i,:] - z * std.numpy()
    upper_t = y_data[i,:] + z * std.numpy()
    time_init1 = np.array([1.0])
    # print(y_test[i,:].shape)
    df_y = pd.DataFrame({"val" : np.transpose(y_test[i,:]),
                   "x" : x})

    df_p = pd.DataFrame({"val" : np.transpose(pred[i,:]).reshape(y_test.shape[1]),
                       #"type" : np.repeat(["obs.","pred."],y_test.shape[1]*y_test.shape[2]),
                       "x" : x})
    df_bound = pd.DataFrame({"lb" : lower.reshape(y_test.shape[1]),
                                "ub" : upper.reshape(y_test.shape[1]),
                       "x" : x})
    df_bound_t = pd.DataFrame({"lb" : lower_t.reshape(y_test.shape[1]),
                                "ub" : upper_t.reshape(y_test.shape[1]),
                       "x" : x})
    # ax = sns.lineplot(x= "x", y="val", data=df, hue = "time",linewidth=2,palette="flare")
    ax = sns.lineplot(x= "x", y="val", data=df_y,linewidth=2, color = "grey", label = "True")
    # ax = sns.lineplot(x= "x", y="val", data=df_p, hue = "time",linestyle = "dotted",linewidth=2,palette="crest_r")
    ax = sns.lineplot(x= "x", y="val", data=df_p, linestyle = "dotted", color = "black", label = "Pred")
    # ax = sns.lineplot(x= x, y=var[i,:], linestyle = "dotted", color = "red", label = "Var")
    #ax = sns.lineplot(x= "x", y="out", data=df, linestyle = "dotted", hue = "time",linewidth=2)
    plt.fill_between(df_bound["x"], df_bound["lb"], df_bound["ub"], color='gray', alpha=0.3, label="90% PI")
    plt.fill_between(df_bound_t["x"], df_bound_t["lb"], df_bound_t["ub"], color='steelblue', alpha=0.3, label="90% PI true")
    ax.set_xlabel('$s$',fontsize=20)
    ax.set_ylabel('$Y_t^{(L)} \quad vs \quad \hat{Y}_t^{(L)}$',fontsize=20)
    # Add title
    # ax.set_title("Instance "+str(i), fontsize=20)

    # Add legend
    ax.legend(loc="upper left")
    ax.set_xticks([]) 
   
    plt.savefig("plots/burger-FNO-nonlcl_dat-lcl-"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    plt.clf()

    ax = sns.lineplot(x= x, y= diff[i,:], linewidth=2, color = "black")
    #ax.set_title("$Y_t^{(L)} \quad - \quad \hat{Y}_t^{(L)}$", fontsize=20)
    ax.set_xlabel('$s$',fontsize=20)
    ax.set_xticks([]) 
    plt.savefig("plots/diff_burger-FNO-nonlcl_dat-lcl-"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    plt.clf()





pred = np.load("pred/burger-ConvLSTM-nonlcl_dat-nonlcl.npy")
var = np.load("pred/burger-ConvLSTM-nonlcl_dat-nonlcl_cov.npy")
# print(var.shape)
y_test = np.load("pred/burger_test_nonlcl.npy")
mse = np.mean((y_test - pred) ** 2)
print("MSE ConvLSTM nonlocal:", mse)
print("MPIW:", np.mean(2*z*var))
# print(y_test.shape)
data = np.load("datasets/generated_1d_data_Burger-beta~U<0.05-0.7>_matern.npy")
y_data = data[-ntest:,:,9]
diff = pred - y_test
#data = np.load("generated_1d_data_Burger-04_matern.npy")
#x_test = data[950:,:,0]
x = np.array(range(2**11))/2**11 - 1
# index = [0,25,45]
plt.figure(figsize=(10, 6))

sns.set(style="whitegrid")
for i in index:
             # shape: [D]
    mean = pred[i, :]                  # shape (D,)
    # cov = var[i, :]                 # shape (D, D)
    # print(cov.shape)
    # sns.heatmap(cov, cmap="viridis", square=True, cbar_kws={'label': 'Covariance'})
    # plt.imshow(cov, cmap='viridis', aspect='auto')
    # plt.colorbar(label="Covariance")
    # plt.title("Covariance Matrix (imshow)")
    # plt.xlabel("Index")
    # plt.ylabel("Index")
    # plt.savefig("plots/burger-FNO-lcl_dat-nonlcl-matrix"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    # plt.clf()
    std =  var[i, :]
    # print(std)
    lower = mean - z * std
    upper = mean + z * std
    x1 = torch.linspace(0, 1, n)

    # RBF Kernel function
    std = 0.08 * torch.ones_like(x1)
    # std = 0.25 * x1 + 1e-2 * torch.ones_like(x1)
    lower_t = y_data[i,:] - z * std.numpy()
    upper_t = y_data[i,:] + z * std.numpy()
    time_init1 = np.array([1.0])
    # print(y_test[i,:].shape)
    df_y = pd.DataFrame({"val" : np.transpose(y_test[i,:]),
                   "x" : x})

    df_p = pd.DataFrame({"val" : np.transpose(pred[i,:]).reshape(y_test.shape[1]),
                       #"type" : np.repeat(["obs.","pred."],y_test.shape[1]*y_test.shape[2]),
                       "x" : x})
    df_bound = pd.DataFrame({"lb" : lower.reshape(y_test.shape[1]),
                                "ub" : upper.reshape(y_test.shape[1]),
                       "x" : x})
    df_bound_t = pd.DataFrame({"lb" : lower_t.reshape(y_test.shape[1]),
                                "ub" : upper_t.reshape(y_test.shape[1]),
                       "x" : x})
    # ax = sns.lineplot(x= "x", y="val", data=df, hue = "time",linewidth=2,palette="flare")
    ax = sns.lineplot(x= "x", y="val", data=df_y,linewidth=2, color = "grey", label = "True")
    # ax = sns.lineplot(x= "x", y="val", data=df_p, hue = "time",linestyle = "dotted",linewidth=2,palette="crest_r")
    ax = sns.lineplot(x= "x", y="val", data=df_p, linestyle = "dotted", color = "black", label = "Pred")
    # ax = sns.lineplot(x= x, y=var[i,:], linestyle = "dotted", color = "red", label = "Var")
    #ax = sns.lineplot(x= "x", y="out", data=df, linestyle = "dotted", hue = "time",linewidth=2)
    plt.fill_between(df_bound["x"], df_bound["lb"], df_bound["ub"], color='gray', alpha=0.3, label="90% PI")
    plt.fill_between(df_bound_t["x"], df_bound_t["lb"], df_bound_t["ub"], color='steelblue', alpha=0.3, label="90% PI true")
    ax.set_xlabel('$s$',fontsize=20)
    ax.set_ylabel('$Y_t^{(L)} \quad vs \quad \hat{Y}_t^{(L)}$',fontsize=20)
    # Add title
    # ax.set_title("Instance "+str(i), fontsize=20)

    # Add legend
    ax.legend(loc="upper left")
    ax.set_xticks([]) 
   
    plt.savefig("plots/burger-ConvLSTM-nonlcl_dat-nonlcl-"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    plt.clf()

    ax = sns.lineplot(x= x, y= diff[i,:], linewidth=2, color = "black")
    #ax.set_title("$Y_t^{(L)} \quad - \quad \hat{Y}_t^{(L)}$", fontsize=20)
    ax.set_xlabel('$s$',fontsize=20)
    ax.set_xticks([]) 
    plt.savefig("plots/diff_burger-ConvLSTM-nonlcl_dat-nonlcl-"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    plt.clf()



pred = np.load("pred/burger-STDK_med-nonlcl_dat-nonlcl.npy")
lb = np.load("pred/burger-STDK_lb-nonlcl_dat-nonlcl.npy")
ub = np.load("pred/burger-STDK_ub-nonlcl_dat-nonlcl.npy")
# print(var.shape)
y_test = np.load("pred/burger_test_nonlcl-FNO-nonlcl.npy")
mse = np.mean((y_test - pred) ** 2)
print("MSE STDK nonlocal:", mse)
print("MPIW:", np.mean(ub-lb))
# print(y_test.shape)
data = np.load("datasets/generated_1d_data_Burger-beta~U<0.05-0.7>_matern.npy")
y_data = data[-ntest:,:,9]
print("persistance : ", np.mean((data[-ntest:,:,4]-y_test)**2))
diff = pred - y_test
#data = np.load("generated_1d_data_Burger-04_matern.npy")
#x_test = data[950:,:,0]
x = np.array(range(2**11))/2**11 - 1
# index = [0,25,45]
plt.figure(figsize=(10, 6))

sns.set(style="whitegrid")
for i in index:
             # shape: [D]
    mean = pred[i, :]                  # shape (D,)
    lbi = lb[i,:]
    ubi = ub[i,:]
    # cov = var[i, :]                 # shape (D, D)
    # print(cov.shape)
    # sns.heatmap(cov, cmap="viridis", square=True, cbar_kws={'label': 'Covariance'})
    # plt.imshow(cov, cmap='viridis', aspect='auto')
    # plt.colorbar(label="Covariance")
    # plt.title("Covariance Matrix (imshow)")
    # plt.xlabel("Index")
    # plt.ylabel("Index")
    # plt.savefig("plots/burger-FNO-lcl_dat-nonlcl-matrix"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    # plt.clf()
    x1 = torch.linspace(0, 1, n)

    # RBF Kernel function
    std = 0.08 * torch.ones_like(x1)
    # std = 0.25 * x1 + 1e-2 * torch.ones_like(x1)
    lower_t = y_data[i,:] - z * std.numpy()
    upper_t = y_data[i,:] + z * std.numpy()
    time_init1 = np.array([1.0])
    # print(y_test[i,:].shape)
    df_y = pd.DataFrame({"val" : np.transpose(y_test[i,:]),
                   "x" : x})

    df_p = pd.DataFrame({"val" : np.transpose(pred[i,:]).reshape(y_test.shape[1]),
                       #"type" : np.repeat(["obs.","pred."],y_test.shape[1]*y_test.shape[2]),
                       "x" : x})
    df_bound = pd.DataFrame({"lb" : lbi.reshape(y_test.shape[1]),
                                "ub" : ubi.reshape(y_test.shape[1]),
                       "x" : x})
    df_bound_t = pd.DataFrame({"lb" : lower_t.reshape(y_test.shape[1]),
                                "ub" : upper_t.reshape(y_test.shape[1]),
                       "x" : x})
    # ax = sns.lineplot(x= "x", y="val", data=df, hue = "time",linewidth=2,palette="flare")
    ax = sns.lineplot(x= "x", y="val", data=df_y,linewidth=2, color = "grey", label = "True")
    # ax = sns.lineplot(x= "x", y="val", data=df_p, hue = "time",linestyle = "dotted",linewidth=2,palette="crest_r")
    ax = sns.lineplot(x= "x", y="val", data=df_p, linestyle = "dotted", color = "black", label = "Pred")
    # ax = sns.lineplot(x= x, y=var[i,:], linestyle = "dotted", color = "red", label = "Var")
    #ax = sns.lineplot(x= "x", y="out", data=df, linestyle = "dotted", hue = "time",linewidth=2)
    plt.fill_between(df_bound["x"], df_bound["lb"], df_bound["ub"], color='gray', alpha=0.3, label="90% PI")
    plt.fill_between(df_bound_t["x"], df_bound_t["lb"], df_bound_t["ub"], color='steelblue', alpha=0.3, label="90% PI true")
    ax.set_xlabel('$s$',fontsize=20)
    ax.set_ylabel('$Y_t^{(L)} \quad vs \quad \hat{Y}_t^{(L)}$',fontsize=20)
    # Add title
    # ax.set_title("Instance "+str(i), fontsize=20)

    # Add legend
    ax.legend(loc="upper left")
    ax.set_xticks([]) 
   
    plt.savefig("plots/burger-STDK-nonlcl_dat-nonlcl-"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    plt.clf()

    ax = sns.lineplot(x= x, y= diff[i,:], linewidth=2, color = "black")
    #ax.set_title("$Y_t^{(L)} \quad - \quad \hat{Y}_t^{(L)}$", fontsize=20)
    ax.set_xlabel('$s$',fontsize=20)
    ax.set_xticks([]) 
    plt.savefig("plots/diff_burger-STDK-nonlcl_dat-nonlcl-"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    plt.clf()



###### plotting the persistance

# print(var.shape)
y_test = np.load("pred/burger_test_nonlcl-FNO-nonlcl.npy")

data = np.load("datasets/generated_1d_data_Burger-beta~U<0.05-0.7>_matern.npy")
data = torch.tensor(data, dtype=torch.float)
n = 2048

# Generate positions (can be 1D spatial points)
x = torch.linspace(0, 1, n)
std = 0.08 * torch.ones_like(x) # shape: (n,)
D = torch.diag(std)
c = rbf_kernel(x.unsqueeze(1), x.unsqueeze(1))

Sigma = D @ c @ D
Sigma += 1e-4 * torch.eye(n)

# Sample from multivariate normal
mean = torch.zeros(n)
mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=Sigma)
samples = mvn.sample((1000,11)).permute(0,2,1)
# print(samples.min())
# print(samples.max())

# print("DATA")
# print(data)
# print("SAMPLES")
# print(samples)
data += samples
# print(data)
data = data.numpy()
pred = data[-ntest:,:,4]
# print(pred.shape)

diff = pred - y_test
#data = np.load("generated_1d_data_Burger-04_matern.npy")
#x_test = data[950:,:,0]
x = np.array(range(2**11))/2**11 - 1
# index = [0,25,45]
plt.figure(figsize=(10, 6))

sns.set(style="whitegrid")
for i in index:
             # shape: [D]
    
    x1 = torch.linspace(0, 1, n)

    # RBF Kernel function
    std = 0.08 * torch.ones_like(x1)
    # std = 0.25 * x1 + 1e-2 * torch.ones_like(x1)
    lower_t = y_data[i,:] - z * std.numpy()
    upper_t = y_data[i,:] + z * std.numpy()
    time_init1 = np.array([1.0])
    # print(y_test[i,:].shape)
    df_y = pd.DataFrame({"val" : np.transpose(y_test[i,:]),
                   "x" : x})

    df_p = pd.DataFrame({"val" : np.transpose(pred[i,:]).reshape(y_test.shape[1]),
                       #"type" : np.repeat(["obs.","pred."],y_test.shape[1]*y_test.shape[2]),
                       "x" : x})
    
    df_bound_t = pd.DataFrame({"lb" : lower_t.reshape(y_test.shape[1]),
                                "ub" : upper_t.reshape(y_test.shape[1]),
                       "x" : x})
    # ax = sns.lineplot(x= "x", y="val", data=df, hue = "time",linewidth=2,palette="flare")
    ax = sns.lineplot(x= "x", y="val", data=df_y,linewidth=2, color = "grey", label = "True")
    # ax = sns.lineplot(x= "x", y="val", data=df_p, hue = "time",linestyle = "dotted",linewidth=2,palette="crest_r")
    ax = sns.lineplot(x= "x", y="val", data=df_p, linestyle="solid", color = "black", label = "Pred")
    # ax = sns.lineplot(x= x, y=var[i,:], linestyle = "dotted", color = "red", label = "Var")
    #ax = sns.lineplot(x= "x", y="out", data=df, linestyle = "dotted", hue = "time",linewidth=2)
    # plt.fill_between(df_bound["x"], df_bound["lb"], df_bound["ub"], color='gray', alpha=0.3, label="90% PI")
    plt.fill_between(df_bound_t["x"], df_bound_t["lb"], df_bound_t["ub"], color='steelblue', alpha=0.3, label="90% PI true")
    ax.set_xlabel('$s$',fontsize=20)
    ax.set_ylabel('$Y_t^{(L)} \quad vs \quad \hat{Y}_t^{(L)}$',fontsize=20)
    # Add title
    # ax.set_title("Instance "+str(i), fontsize=20)

    # Add legend
    ax.legend(loc="upper left")
    ax.set_xticks([]) 
   
    plt.savefig("plots/burger-persistance-nonlcl_"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    plt.clf()

    ax = sns.lineplot(x= x, y= diff[i,:], linewidth=2, color = "black")
    #ax.set_title("$Y_t^{(L)} \quad - \quad \hat{Y}_t^{(L)}$", fontsize=20)
    ax.set_xlabel('$s$',fontsize=20)
    ax.set_xticks([]) 
    plt.savefig("plots/diff_burger-burger-persistance-nonlcl_"+str(i)+".pdf", format='pdf',dpi=600, bbox_inches='tight')
    plt.clf()