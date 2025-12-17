#!/usr/bin/env python
# coding: utf-8

"""
Created on Friday January 24 13:33:04 2025

@author: Pratik
"""
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

def k(distsq, D, dt):
    return torch.exp(-distsq / (4 * D * dt)) / (4 * np.pi * D * dt)

def gradientk(dist, D, dt):
    # print(dist.shape)
    term1 = torch.exp(-(dist**2).sum(1) / (4 * D * dt)) / (8 * np.pi * D**2 * dt**2)
    term1 = term1.reshape(dist.shape[0],1,dist.shape[2],dist.shape[3],dist.shape[4],dist.shape[5])
    # print(term1.shape)
    return dist * term1

class GaussianConvolution(Function):
    D = 0.45
    dt = 1

    @staticmethod
    def forward(ctx, w, I):
        ctx.save_for_backward(w, I)
        interval = torch.arange(I.size()[-1], device=I.device, dtype=I.dtype)
        x1 = interval[None, :, None, None, None]
        x2 = interval[None, None, :, None, None]
        y1 = interval[None, None, None, :, None]
        y2 = interval[None, None, None, None, :]
        distsq = (x1 - y1 - w[:, 0, :, :, None, None]) ** 2 + (x2 - y2 - w[:, 1, :, :, None, None]) ** 2
        return (I[:, None, None, :, :] * k(distsq, GaussianConvolution.D, GaussianConvolution.dt)).sum(4).sum(3)

    @staticmethod
    def backward(ctx, grad_output):
        w, I = ctx.saved_tensors
        interval = torch.arange(I.size()[-1], device=I.device, dtype=I.dtype)
        x1 = interval[None, :, None, None, None]
        x2 = interval[None, None, :, None, None]
        y1 = interval[None, None, None, :, None]
        y2 = interval[None, None, None, None, :]
        distx = (x1 - w[:, 0, :, :, None, None] - y1)[:, None, :, :, :, :].repeat(1, 1, 1, 1, 1, I.size()[-1])
        disty = (x2 - w[:, 1, :, :, None, None] - y2)[:, None, :, :, :, :].repeat(1, 1, 1, 1, I.size()[-1], 1)
        dist = torch.cat((distx, disty), dim=1)
        # print(I.shape)
        # print(gradientk(dist, GaussianConvolution.D, GaussianConvolution.dt).shape)
        grad = (I[:, None, None, None, :, :] * gradientk(dist, GaussianConvolution.D, GaussianConvolution.dt)).sum(5).sum(4)
        return grad * grad_output[:, None, :, :], None

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_normal_(module.weight, gain=1)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_EncoderBlock, self).__init__()
        self.cv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.lr1 = nn.LeakyReLU(0.1, inplace=False)  # Fixed in-place issue
        self.cv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lr2 = nn.LeakyReLU(0.1, inplace=False)  # Fixed in-place issue
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=1)

    def forward(self, x):
        residual = x
        out = self.cv1(x)
        out = self.bn1(out)
        out = self.lr1(out)
        out += residual
        out = self.cv2(out)
        out = self.bn2(out)
        out = self.lr2(out)
        out = self.maxp(out)
        return out

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.cv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.lr1 = nn.LeakyReLU(0.1, inplace=False)  # Fixed in-place issue
        self.cv2 = nn.Conv2d(in_channels, middle_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.lr2 = nn.LeakyReLU(0.1, inplace=False)  # Fixed in-place issue
        self.tcv = nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3)

    def forward(self, x):
        residual = x
        out = self.cv1(x)
        out = self.bn1(out)
        out = self.lr1(out)
        out += residual
        out = self.cv2(out)
        out = self.bn2(out)
        out = self.lr2(out)
        out = self.tcv(out)
        return out

class CDNN(nn.Module):
    def __init__(self, k):
        super(CDNN, self).__init__()
        self.enc1 = _EncoderBlock(k, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512)
        self.dec4 = _DecoderBlock(512, 386, 256)
        self.dec3 = _DecoderBlock(256 + 256, 256, 194)
        self.dec2 = _DecoderBlock(194 + 128, 128, 98)
        self.dec1 = _DecoderBlock(98 + 64, 64, 2)
        self.final = nn.Conv2d(2, 2, kernel_size=3)

        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        dec4 = self.dec4(enc4)
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, size=dec4.shape[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, size=dec3.shape[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, size=dec2.shape[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        W = F.interpolate(final, size=x.shape[2:], mode='bilinear')

        return W