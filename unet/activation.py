import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

class Activation(nn.Module):
    def __init__(self, act = 'esh'):
        super(Activation, self).__init__()
        self.act = None
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "gelu":
            self.act = nn.GELU()    
        elif act == "selu":
            self.act = nn.SELU() 
        elif act == "leakyrelu":
            self.act = nn.LeakyReLU() 
        elif act == "elu":
            self.act = nn.ELU() 
        elif act == "mish":
            self.act = nn.Mish()
        elif act == "prelu":
            self.act = nn.PReLU()
        elif act == "esh":
            self.act = Esh()
        else:
            self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(x)

class Esh(nn.Module):
    def __init__(self):
        super(Esh, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.sigmoid(x))
    