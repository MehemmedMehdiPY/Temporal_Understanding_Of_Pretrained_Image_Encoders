import torch
from torch import nn
from .convlstm import ConvLSTM
import torch.nn.functional as f

class FramePredictor(nn.Module):
    def __init__(self, input_size, kernel_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.backbone = ConvLSTM(input_size=input_size, hidden_size=hidden_size, kernel_size=kernel_size, num_layers=3)
        self.conv = nn.Conv2d(in_channels=hidden_size, out_channels=input_size[0], kernel_size=kernel_size, padding="same")
        
    def forward(self, x):
        o, _ = self.backbone(x)
        o = self.conv(o)
        o = f.sigmoid(o)
        return o
