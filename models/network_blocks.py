import os, sys, time
import math
import numpy as np

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=4, stride=2, padding=1,
                 bias=False, norm_layer=None, activation_fn = nn.LeakyReLU(0.2, True)):
        super(ConvBlock, self).__init__()
        self.conv_block = self.build_conv_block(ch_in, ch_out, kernel_size,
                                                stride, padding, bias, norm_layer, activation_fn)
    
    def build_conv_block(self, ch_in, ch_out, kernel_size, stride,
                         padding, bias, norm_layer, activation_fn):
        conv_block = list()
        conv_block += [nn.Conv2d(in_channels = ch_in,
                                 out_channels = ch_out,
                                 kernel_size = kernel_size,
                                 stride = stride,
                                 padding = padding,
                                 bias = bias)]
        conv_block += [activation_fn]
        if norm_layer is not None:
            conv_block += [norm_layer(ch_out)]
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        return self.conv_block(x)


class ConvTransBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=4, stride=2, padding=1,
                 bias=False, norm_layer=None, activation_fn = nn.LeakyReLU(0.2, True)):
        super(ConvTransBlock, self).__init__()
        self.convtrans_block = self.build_convtrans_block(ch_in, ch_out, kernel_size, stride,
                                                          padding, bias, norm_layer, activation_fn)

    def build_convtrans_block(self, ch_in, ch_out, kernel_size, stride,
                              padding,bias, norm_layer, activation_fn):
        convtrans_block = list()
        convtrans_block += [nn.ConvTranspose2d(in_channels = ch_in,
                                               out_channels = ch_out,
                                               kernel_size = kernel_size,
                                               stride = stride,
                                               padding = padding,
                                               bias = bias)]
        convtrans_block += [activation_fn]
        if norm_layer is not None:
            convtrans_block += [norm_layer(ch_out)]
        return nn.Sequential(*convtrans_block)
    
    def forward(self, x):
        return self.convtrans_block(x)


class VGG_layer(nn.Module):
    
    def __init__(self, ch_in, ch_out):
        layer = []
        layer += [nn.Conv2d(in_channels = ch_in, out_channels = ch_out, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ch_out), nn.LeakyReLU(0.2, True)]
        self.layer = nn.Sequential(*layer)
    
    def forward(self, x):
        return self.layer(x)



## Codes are from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels = in_planes,
                     out_channels = out_planes,
                     kernel_size = 3,
                     stride = stride,
                     padding = 1,
                     bias = False)


## Codes are from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
## This class is only used in resnet18, resnet34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1      = conv3x3(inplanes, planes, stride)
        self.bn1        = nn.BatchNorm2d(planes)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = conv3x3(planes, planes)
        self.bn2        = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


## Codes are from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
## This class is only used in resnet50, resnet101, resnet152
## Not in resnet18 which we do not used
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes * self.expansion)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
