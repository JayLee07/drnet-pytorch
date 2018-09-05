import math
import torch
import torch.nn as nn

from model_utils import initialize_weights, get_norm_layer
from network_blocks import ConvBlock, ConvTransBlock


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