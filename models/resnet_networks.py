import torch.nn as nn
import math


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


## Codes are from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class ResNet(nn.Module):

    def __init__(self, args):
        super(ResNet, self).__init__()
        block         = BasicBlock
        layers        = [2,2,2,2,2]
        self.args = args
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.relu     = nn.ReLU(inplace=True)
        self.maxpool  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1   = self._make_layer(block, 64,   layers[0])
        self.layer2   = self._make_layer(block, 128,  layers[1], stride=2)
        self.layer3   = self._make_layer(block, 256,  layers[2], stride=2)
        self.layer4   = self._make_layer(block, 512,  layers[3], stride=2)
        self.layer5   = self._make_layer(block, 1024, layers[4], stride=2)
        self.bn_out   = nn.BatchNorm2d(args.pose_dim)
        self.tanh     = nn.Tanh()
        
        if args.width == 64:
            self.conv_out = nn.Conv2d(512, args.pose_dim, kernel_size=3)
        elif args.width == 128:
            self.conv_out = nn.Conv2d(1024, args.pose_dim, kernel_size=3)

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