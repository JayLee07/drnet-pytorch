import torch
import torch.nn as nn

class Scene_discriminator(nn.Module):
    
    def __init__(self, args):
        super(Scene_discriminator, self).__init__()
        self.args = args
        self.layers = nn.Sequential(nn.Linear(self.args.pose_dim * 2, self.args.sceneD_hidden_layers),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(args.sceneD_hidden_layers, args.sceneD_hidden_layers),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(args.sceneD_hidden_layers, 1),
                                    nn.Sigmoid())
    
    def forward(self, x):
        output = self.layers(torch.cat(x, 1).view(-1, self.args.pose_dim * 2))
        return output