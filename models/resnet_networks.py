import torch.nn as nn
import math
from network_blocks import conv3x3, BasicBlock, Bottleneck



## Codes are from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class ResNet18_Encoder(nn.Module):

    def __init__(self, args):
        super(ResNet18_Encoder, self).__init__()
        block         = BasicBlock
        layers        = [2,2,2,2,2]
        self.args = args
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, self.args.ngf, kernel_size=5, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.relu     = nn.ReLU(inplace=True)
        self.maxpool  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1   = self._make_layer(block, self.args.ngf,    layers[0])
        self.layer2   = self._make_layer(block, self.args.ngf*2,  layers[1], stride=2)
        self.layer3   = self._make_layer(block, self.args.ngf*4,  layers[2], stride=2)
        self.layer4   = self._make_layer(block, self.args.ngf*8,  layers[3], stride=2)
        self.layer5   = self._make_layer(block, self.args.ngf*16, layers[4], stride=2)
        self.bn_out   = nn.BatchNorm2d(args.pose_dim)
        self.tanh     = nn.Tanh()
        
        if args.width == 64:
            self.conv_out = nn.Conv2d(self.args.ngf*8, args.pose_dim, kernel_size=3)
        elif args.width == 128:
            self.conv_out = nn.Conv2d(self.args.ngf*16, args.pose_dim, kernel_size=3)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.args.width == 128:
            x = self.layer5(x)

        x = self.conv_out(x)
        x = self.bn_out(x)
        x = self.tanh(x)
        return x