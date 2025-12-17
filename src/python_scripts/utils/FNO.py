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

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utils.utilities3 import *


torch.manual_seed(0)
np.random.seed(0)


################################################################
#  1d FNO
################################################################

#Complex multiplication
def compl_mul1d(a, b):
    # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
    op = partial(torch.einsum, "bix,iox->box")
    
    real_part = op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1])
    imag_part = op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    
    return torch.stack([real_part, imag_part], dim=-1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1


        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, dim = -1, norm="ortho")
        
        x_ft = torch.stack([x_ft[:, :, :self.modes1].real,x_ft[:, :, :self.modes1].imag], dim = -1)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1] = compl_mul1d(x_ft, self.weights1)
        out_ft = torch.view_as_complex(out_ft)
        #Return to physical space
        x = torch.fft.irfft(out_ft, n = x.size(-1), norm="ortho" )
        return x

class SimpleBlock1d(nn.Module):
    def __init__(self, modes, width):
        super(SimpleBlock1d, self).__init__()

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm1d(self.width)
        self.bn1 = torch.nn.BatchNorm1d(self.width)
        self.bn2 = torch.nn.BatchNorm1d(self.width)
        self.bn3 = torch.nn.BatchNorm1d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        # self.fc3 = nn.Linear(128, 1)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = self.bn3(x1 + x2)


        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x1 = self.fc2(x)
        # x2 = F.softplus(self.fc3(x))
        return x1

class Net1d(nn.Module):
    def __init__(self, modes, width):
        super(Net1d, self).__init__()

        self.conv1 = SimpleBlock1d(modes, width)


    def forward(self, x):
        x1 = self.conv1(x)
        return x1.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
        
 
 
#################################################################
################ 2D FNO   #######################################
#################################################################
 
#Complex multiplication
def compl_mul2d(a, b):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    op = partial(torch.einsum, "bixy,ioxy->boxy")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)



class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        #self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Compute Fourier coefficients using torch.fft
        x_ft = torch.fft.rfftn(x, dim=(-2, -1), norm="ortho") #
        x_ft = torch.stack([x_ft[:, :, :self.modes1].real,x_ft[:, :, :self.modes1].imag], dim = -1)
        
        # Prepare output tensor for relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-2)// 2 + 1, x.size(-1) // 2 + 1,2, device=x.device)

        # Multiply relevant Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        #out_ft[:, :, -self.modes1:, :self.modes2] = \
            #compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft = torch.view_as_complex(out_ft)
        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho") #s=(x.size(-2), x.size(-1)), dim=(-2, -1),
        #print(x.shape)
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(3, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(5,1)    ### layer for varied time poutput

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]
        
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn3(x1 + x2)


        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        ## for getting output at a separate time length
        x = x.permute(0,1,3,2)
        x = self.fc3(x)
        return x

class Net2d(nn.Module):
    def __init__(self, modes1,modes2, width):
        super(Net2d, self).__init__()

        self.conv1 = SimpleBlock2d(modes1, modes2,  width)


    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
        
        
        
################################################################
# 3D FNO
################################################################

def compl_mul3d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    op = partial(torch.einsum, "bixyz,ioxyz->boxyz")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

class SpectralConv3d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        # self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        # self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        # self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")  # Using rfftn for real-to-complex FFT
        x_ft = torch.stack([x_ft.real,x_ft.imag], dim = -1)
        
        # # Prepare output tensor for relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-3)// 2 +1, x.size(-2)// 2 + 1, x.size(-1) // 2 + 1,2, device=x.device)
       
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        # out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
        #     compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        # out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
        #     compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        # out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
        #     compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        # print("I am here2")
        # Convert back to physical space
        out_ft = torch.view_as_complex(out_ft)

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)), norm="ortho")  # Using irfftn for inverse FFT
        return x

class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(SimpleBlock3d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(4, self.width)

        self.conv0 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(3,1)    ### layer for varied time poutput
        # Trainable parameter for x_
        # self.x_ = nn.Parameter(torch.tensor(0.5))  # initialize at some value (e.g., 0.5)
        
        # Final output layer for x_
        # self.fc_x_ = nn.Linear(128, 1)

    def forward(self, x):
        # print("I am here 1")
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        # print("I am here 2")
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn3(x1 + x2)
        

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.squeeze()
        # x = x.permute(0,1,3,2)
        x = self.fc3(x)
        x = x.squeeze(-1)
        return x

class Net3d(nn.Module):
    def __init__(self, modes1,modes2,modes3, width):
        super(Net3d, self).__init__()

        self.conv1 = SimpleBlock3d(modes1, modes2, modes3, width)


    def forward(self, x):
        x = self.conv1(x)
        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            print(p)
            c += reduce(operator.mul, list(p.size()))
        return c


