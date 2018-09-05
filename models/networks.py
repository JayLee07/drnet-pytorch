import os, sys, time
import math
import numpy as np

import torch
import torch.nn as nn
from model_utils import initialize_weights, get_norm_layer
from resnet_networks import BasicBlock


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


class DCGAN_Encoder(nn.Module):
    def __init__(self, args, normalize=False, gpu_ids=[]):
        super(DCGAN_Encoder).__init__()
        self.args = args
        self.normalize = normalize
        self.gpu_ids = gpu_ids
        norm_layer = get_norm_layer(args.norm_type)
        n_downsampling = int(math.log2(args.width))-3
        layer = list()
        layer += [ConvBlock(ch_in = args.ch_in, ch_out = args.ngf, norm_layer = norm_layer)]
        for i in range(n_downsampling):
            layer += [ConvBlock(ch_in = args.ngf * (2**i),
                                ch_out = args.ngf * (2**(i+1)),
                                norm_layer = norm_layer)]
        layer += [ConvBlock(ch_in = args.ngf * (2**n_downsampling),
                            ch_out = args.pose_dim,
                            kernel_size = 4,
                            stride = 1,
                            padding = 0,
                            bias = False,
                            norm_layer = norm_layer,
                            activation_fn = nn.Tanh())]
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        if self.normalize:
            return nn.functional.normalize(self.layer(x), p = 2)
        else:
            return self.layer(x)


class DCGAN_Decoder(nn.Module):
    def __init__(self, args, gpu_ids=[]):
        super(DCGAN_Decoder).__init__()
        self.args = args
        self.gpu_ids = gpu_ids
        norm_layer = get_norm_layer(args.norm_type)
        n_upsampling = int(math.log2(args.width))-3
        layer = list()
        layer += [ConvTransBlock(ch_in = (args.content_dim + args.pose_dim),
                                 ch_out = args.ngf * (2**n_upsampling),
                                 kernel_size = 4,
                                 stride = 1,
                                 padding = 0,
                                 bias = False,
                                 norm_layer = norm_layer)]
        for i in range(n_upsampling, 0, -1):
            layer += [ConvTransBlock(ch_in = args.ngf * (2**i),
                                     ch_out = args.ngf * (2**(i-1)),
                                     norm_layer = norm_layer)]
        layer += [ConvTransBlock(ch_in = args.ngf,
                                 ch_out = args.ch_out,
                                 activation_fn = nn.Sigmoid())]
        self.layer = nn.Sequential(*layer)
    
    def forward(self, x):
        content, pose = x
        x_cat = torch.cat([content, pose], 1)
        return self.layer(x_cat)
    
        
class Unet_Encoder(nn.Module):
    def __init__(self, args, gpu_ids=[]):
        super(Unet_Encoder).__init__()
        self.args = args
        self.gpu_ids = gpu_ids
        norm_layer = get_norm_layer(args.norm_type)
        n_downsampling = int(math.log2(args.width))-3
        layer = list()
        layer += [ConvBlock(ch_in = args.ch_in, ch_out = args.ngf)]
        for i in range(n_downsampling):
            layer += [ConvBlock(ch_in = args.ngf * (2**i), ch_out = args.ngf * (2 ** (i+1)))]
        layer += [ConvBlock(ch_in = args.ngf * (2**n_downsampling),
                            ch_out = args.pose_dim,
                            kernel_size = 4,
                            stride = 1,
                            padding = 0,
                            bias = False,
                            norm_layer = norm_layer,
                            activation_fn = nn.Tanh())]
        self.layer = layer
        
    def forward(self, x):
        skips = []
        for i in range(len(self.layer)):
            skips.append(self.layer[i](x))
        return skips[-1], skips[0:-1]
    
    
class Unet_Decoder(nn.Module):
    def __init__(self, args, gpu_ids=[]):
        super(Unet_Decoder).__init__()
        self.args = args
        self.gpu_ids = gpu_ids
        norm_layer = get_norm_layer(args.norm_type)
        n_upsampling = int(math.log2(args.width))-3
        layer = list()
        layer += [ConvTransBlock(ch_in = (args.content_dim + args.pose_dim),
                                 ch_out = args.ngf * (2**n_upsampling),
                                 kernel_size = 4,
                                 stride = 1,
                                 padding = 0,
                                 bias = False,
                                 norm_layer = norm_layer)]
        for i in range(n_upsampling, 0, -1):
            layer += [ConvTransBlock(ch_in = args.ngf * 2 * (2**i),
                                     ch_out = args.ngf * (2**(i-1)),
                                     norm_layer = norm_layer)]
        layer += [ConvTransBlock(ch_in = args.ngf * 2,
                                 ch_out = args.ch_in,
                                 activation_fn = nn.Sigmoid())]
        self.layer = layer
        
    def forward(self, x):
        content, pose = x
        content, skip = content
        x = torch.cat([content, pose], 1)
        h = self.layer[0](x)
        skip = list(reversed(skip))
        for i in range(0, len(self.layer)-1):
            h = self.layer[i+1](torch.cat([h, skip[i]],1))
        return h


class ResNet18_Encoder(nn.Module):

    def __init__(self, args, block=BasicBlock):
        self.inplanes = 64
        self.layers = args.layers
        assert 2 ** (args.layers+2) == args.width
        super(ResNet18_Encoder, self).__init__()
        layer =[]
        layer += [nn.Conv2d(in_channels = args.ch_in,
                            out_channels=args.ngf,
                            kernel_size=5,
                            stride=2,
                            padding=3,
                            bias=False)]
        layer += [nn.BatchNorm2d(args.ngf), nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        layer += [self._make_layer(block, args.ngf, args.layers[0])]
        
        for i in range(1, len(args.layers)):
            layer += [self._make_layer(block, args.ngf * (2**i), args.layers[i], stride=2)]
        
        layer += [nn.Conv2d(in_channels = args.ngf * (2**i),
                            out_channels = args.pose_dim,
                            kernel_size=3)]
        layer += [nn.BatchNorm2d(args.pose_dim), nn.Tanh()]
        self.layer = nn.Sequential(*layer)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        block_layers = []
        block_layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            block_layers.append(block(self.inplanes, planes))
        return block_layers

        # return nn.Sequential(*block_layers)

    def forward(self, x):
        return self.layer(x)