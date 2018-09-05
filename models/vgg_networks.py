import torch
import torch.nn as nn
import math

class VGG_layer(nn.Module):
    
    def __init__(self, ch_in, ch_out):
        layer = []
        layer += [nn.Conv2d(in_channels = ch_in, out_channels = ch_out, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ch_out), nn.LeakyReLU(0.2, True)]
        self.layer = nn.Sequential(*layer)
    
    def forward(self, x):
        return self.layer(x)


class VGG_Unet_Encoder(nn.Module):

    def __init__(self, args):
        super(VGG_Unet_Encoder, self).__init__()
        self.args = args
        self.h1     = nn.Sequential(VGG_layer(args.ch_in, args.ngf),   VGG_layer(args.ngf, args.ngf))
        self.h2     = nn.Sequential(VGG_layer(args.ngf,   args.ngf*2), VGG_layer(args.ngf*2, args.ngf*2))
        self.h3     = nn.Sequential(VGG_layer(args.ngf*2, args.ngf*4), VGG_layer(args.ngf*4, args.ngf*4), VGG_layer(args.ngf*4, args.ngf*4))
        self.h4     = nn.Sequential(VGG_layer(args.ngf*4, args.ngf*8), VGG_layer(args.ngf*8, args.ngf*8), VGG_layer(args.ngf*8, args.ngf*8))
        self.h4_0   = nn.Sequential(VGG_layer(args.ngf*8, args.ngf*8), VGG_layer(args.ngf*8, args.ngf*8), VGG_layer(args.ngf*8, args.ngf*8))
        self.embed  = nn.Sequential(nn.Conv2d(args.ngf*8, args.content_dim, kernel_size=4, stride=1, padding=0), nn.BatchNorm2d(args.ngf*8), nn.Tanh())
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        h1 = self.h1(x)
        h2 = self.h2(self.pool(h1))
        h3 = self.h3(self.pool(h2))
        h4 = self.h4(self.pool(h3))
        if self.args.width == 64:
            embed = self.embed(self.pool(h4))
            return embed, [h1, h2, h3, h4]
        elif self.args.width == 128:
            h5 = self.h4_0(self.pool(h4))
            embed = self.embed(self.pool(h5))
            return embed, [h1, h2, h3, h4, h5]


class VGG_Unet_Decoder(nn.Module):
    
    def __init__(self, args):
        super(VGG_Unet_Decoder, self).__init__()
        self.args = args
        self.h1   = nn.Sequential(nn.ConvTranspose2d(in_channels = args.content_dim + args.pose_dim, out_channels=args.ngf*8, kernel_size=4, stride=1, padding=0), nn.BatchNorm2d(args.ngf*8), nn.LeakyReLU(0.2, inplace=True))
        self.h2_0 = nn.Sequential(VGG_layer(args.ngf*8*2, args.ngf*8), VGG_layer(args.ngf*8, args.ngf*8), VGG_layer(args.ngf*8, args.ngf*8))
        self.h2   = nn.Sequential(VGG_layer(args.ngf*8*2, args.ngf*8), VGG_layer(args.ngf*8, args.ngf*8), VGG_layer(args.ngf*8, args.ngf*4))
        self.h3   = nn.Sequential(VGG_layer(args.ngf*4*2, args.ngf*4), VGG_layer(args.ngf*4, args.ngf*4), VGG_layer(args.ngf*4, args.ngf*2))
        self.h4   = nn.Sequential(VGG_layer(args.ngf*2*2, args.ngf*2), VGG_layer(args.ngf*2, args.ngf))
        self.h5   = nn.Sequential(VGG_layer(args.ngf*2, args.ngf), nn.ConvTranspose2d(in_channels = args.ngf, out_channels = args.ch_in, kernel_size=3, stride=1, padding=1), nn.Sigmoid())
        self.up   = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        content, pose = x
        content, skip = content
        
        if self.args.width == 128:
            x = self.h1(torch.cat([content, pose], 1))
            x = self.h2_0(self.up(torch.cat([x, skip[4], 1)))
            x = self.h2(self.up(torch.cat([x, skip[3], 1)))
            x = self.h3(self.up(torch.cat([x, skip[2], 1)))
            x = self.h4(self.up(torch.cat([x, skip[1], 1)))
            output = self.h5(self.up(torch.cat([x, skip[0], 1)))
        elif self.args.width == 64:
            x = self.h1(torch.cat([content, pose], 1))
            x = self.h2(self.up(torch.cat([x, skip[3], 1)))
            x = self.h3(self.up(torch.cat([x, skip[2], 1)))
            x = self.h4(self.up(torch.cat([x, skip[1], 1)))
            output = self.h5(self.up(torch.cat([x, skip[0], 1)))
        return output