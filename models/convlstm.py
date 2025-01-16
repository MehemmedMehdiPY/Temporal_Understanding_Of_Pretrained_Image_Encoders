"""
ConvLSTM architecture is based on the paper 
        Xingjian S., et al "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
        available at https://arxiv.org/abs/1506.04214
"""

import torch
from torch import nn
import torch.nn.functional as f

class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, kernel_size, hidden_size):
        super().__init__()

        c = input_size[0]
        shape = input_size[1:]

        self.conv_ii = nn.Conv2d(in_channels=c, out_channels=hidden_size, kernel_size=kernel_size, padding="same")
        self.conv_hi = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding="same")
        self.W_ci = nn.Parameter(torch.Tensor(hidden_size, *shape))
        self.b_i = nn.Parameter(torch.Tensor(1, hidden_size, 1, 1))

        self.conv_if = nn.Conv2d(in_channels=c, out_channels=hidden_size, kernel_size=kernel_size, padding="same")
        self.conv_hf = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding="same")
        self.W_cf = nn.Parameter(torch.Tensor(hidden_size, *shape))
        self.b_f = nn.Parameter(torch.Tensor(1, hidden_size, 1, 1))
        
        self.conv_ic = nn.Conv2d(in_channels=c, out_channels=hidden_size, kernel_size=kernel_size, padding="same")
        self.conv_hc = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding="same")
        self.b_c = nn.Parameter(torch.Tensor(1, hidden_size, 1, 1))
        
        self.conv_io = nn.Conv2d(in_channels=c, out_channels=hidden_size, kernel_size=kernel_size, padding="same")
        self.conv_ho = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding="same")
        self.W_co = nn.Parameter(torch.Tensor(hidden_size, *shape))
        self.b_o = nn.Parameter(torch.Tensor(1, hidden_size, 1, 1))

        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            nn.init.normal_(param)
    
    def forward(self, x, previous_gate):
        hidden_gate, cell_gate = previous_gate

        input_gate = f.sigmoid(
            self.conv_ii(x) + 
            self.conv_hi(hidden_gate) + 
            (cell_gate * self.W_ci) + 
            self.b_i
            )
        forget_gate = f.sigmoid(
            self.conv_if(x) + 
            self.conv_hf(hidden_gate) + 
            (cell_gate * self.W_cf) + 
            self.b_f
            )
        cell_gate = (
            forget_gate * cell_gate + 
            input_gate * f.tanh(
                self.conv_ic(x) + 
                self.conv_hc(hidden_gate) + 
                self.b_c
            )
        )
        output_gate = f.sigmoid(
            self.conv_io(x) + 
            self.conv_ho(hidden_gate) + 
            self.W_co * cell_gate + 
            self.b_o
            )
        hidden_gate = output_gate * f.tanh(cell_gate)
        return output_gate, (hidden_gate, cell_gate)

class ConvLSTM(nn.Module):
    def __init__(self, input_size, kernel_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = ConvLSTMCell(input_size=input_size, kernel_size=kernel_size, hidden_size=hidden_size)
    
    def forward(self, x):
        batch_size, seq_size, _, _, _ = x.size()
        hidden_gate = torch.zeros(batch_size, self.hidden_size, *self.input_size[1:])
        cell_gate = torch.zeros(batch_size, self.hidden_size, *self.input_size[1:])

        for idx in range(seq_size):
            x_input = x[:, idx, :, :, :]
            output_gate, (hidden_gate, cell_gate) = self.cell(x_input, (hidden_gate, cell_gate))
        return output_gate, (hidden_gate, cell_gate)
    
if __name__ == "__main__":
    batch_size = 16
    hidden_size = 32
    kernel_size = 3
    input_size = (3, 256, 256)

    x = torch.empty(size=(batch_size, 8, *input_size))
    print(x.shape)
    hidden_gate = torch.zeros(batch_size, hidden_size, *input_size[1:])
    cell_gate = torch.zeros(batch_size, hidden_size, *input_size[1:])
    model = ConvLSTM(input_size=input_size, kernel_size=kernel_size, hidden_size=hidden_size)

    o, (h, c) = model(x)
    print(o.shape, h.shape, c.shape)
    print(model)