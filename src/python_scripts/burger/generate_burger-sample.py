#!/usr/bin/env python
# coding: utf-8

"""
Created on Friday January 24 13:33:04 2025

@author: Pratik
"""

import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern

#import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utilities3 import *
from scipy.spatial import distance_matrix
from tqdm import tqdm 

# exponential covariance function
## define the Matern kernel with parameters 
def chordal_Matern(nx):
    x = np.linspace(0, 2 * np.pi-0.01, nx)
    x_cart = np.zeros((2,nx))
    x_cart[0,:] = np.cos(x)
    x_cart[1,:] = np.sin(x)
    dist_mat = np.matmul(np.transpose(x_cart),x_cart)
    K = np.zeros((nx,nx))
    count = 0
    for i in range(nx):
        for j in range(nx):
            if dist_mat[i,j]>1:
                dist_mat[i,j] = 1.0
            elif dist_mat[i,j]<-1:
                dist_mat[i,j] = -1.0
            cdist = 2*np.sin(math.acos(dist_mat[i,j])/2)
            if cdist < 0:
                count +=1 
            K[i,j]= Matern(length_scale= 1.0, nu=2)(0, cdist.ravel()[:, np.newaxis]).reshape(cdist.shape)
    # cdist = 2*np.sin(math.acos(dist_mat)/2)
    # K = Matern()(0, cdist.ravel()[:, np.newaxis]).reshape(cdist.shape)
    return K

def cholsky(nx):
    K = chordal_Matern(nx)
    return np.linalg.cholesky(K)
    
def r_gp_matern(nx,L):
    # L = np.linalg.inv(cov_mat)
    x_ = np.random.normal(0,1,nx)
    return np.matmul(L,x_)


# solve burgers equation 

def burgers_eqn(nx,nt,nu,u):
    dx = 2 * np.pi / (nx - 1)
    dt = 1/nt
    x = np.linspace(0, 2 * np.pi, nx)
    un = np.empty(nx)
    un1 = np.empty([nx,11])
    un1[:,0] = u.copy()
    t = 0
    for n in range(nt):
        un = u.copy()
        if n>0 and n%50000==0:
            un1[:,(n//50000)] = un.copy()
        # Vectorized update for the internal points (i from 1 to nx-2)
        u[1:-1] = un[1:-1] - dt / dx * un[1:-1] * (un[1:-1] - un[0:-2]) \
                    + nu * dt / dx**2 * (un[2:] - 2 * un[1:-1] + un[0:-2])
        
        # Boundary condition at u[0] (using periodic boundary conditions)
        u[0] = un[0] - dt / dx * un[0] * (un[0] - un[-2]) \
                + nu * dt / dx**2 * (un[1] - 2 * un[0] + un[-2])
        
        # Apply periodic boundary condition
        u[-1] = u[0]
    un1[:,10] = u.copy()
    return un1
    
####################################################################################################################### 
################## Data generation ####################################################################################
#######################################################################################################################

folder_path = "datasets"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Created folder: {folder_path}")
else:
    print(f"Folder already exists: {folder_path}")
    
folder_path = "pred"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Created folder: {folder_path}")
else:
    print(f"Folder already exists: {folder_path}")
    
folder_path = "plots"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Created folder: {folder_path}")
else:
    print(f"Folder already exists: {folder_path}")

nx = 2**11
nt = 500000

data_size = 1000
data_all = []
filename = "datasets/generated_cholsky_2^11-1-2.npy"

if os.path.exists(filename):
    L = np.load(filename)
    print("Loaded existing Cholesky matrix from file.")
else:
    L = cholsky(nx)
    np.save(filename, L)
    print("Generated and saved new Cholesky matrix.")

###################################### 
### for varying beta 
######################################

beta = np.random.uniform(0.05,0.7,data_size)

for i in tqdm(range(data_size)):
    init_u = r_gp_matern(nx,L)
    un1 = burgers_eqn(nx,nt,beta[i],init_u)
    data_all.append(un1)
data_all = np.array(data_all)

np.save("datasets/generated_1d_data_Burger-beta~U<0.05-0.7>_matern.npy",data_all)

###################################### 
### for fixed beta 
######################################

data_all = []
beta = 0.4

for i in tqdm(range(data_size)):
    init_u = r_gp_matern(nx,L)
    un1 = burgers_eqn(nx,nt,beta,init_u)
    data_all.append(un1)
data_all = np.array(data_all)

np.save("datasets/generated_1d_data_Burger-04_matern.npy",data_all)
