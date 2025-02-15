"""
ConvLSTM architecture is based on the paper 
        Xingjian S., et al "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
        available at https://arxiv.org/abs/1506.04214
"""

import torch
from torch import nn
import torch.nn.functional as f
from typing import List, Tuple, Optional, Union

class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, kernel_size, hidden_size, first_layer: Optional[bool] = True, last_layer: Optional[bool] = True):
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

        self.hidden_size = hidden_size
        self.forward = (
            self.forward_first_layer if (first_layer and not last_layer)
            else self.forward_intermediate_layer if (not first_layer and not last_layer)
            else self.forward_last_layer if (not first_layer and last_layer)
            else self.forward_one_layer
        )
        
    def init_weights(self):
        for param in self.parameters():
            nn.init.normal_(param)
    
    def forward_one_layer(self, input):
        x, (hidden_gate, cell_gate) = input
        _, seq_size, _, _, _ = x.size()
        for idx in range(seq_size):
            x_input = x[:, idx, :, :, :]
            output_gate, (hidden_gate, cell_gate) = self.forward_cell(
                (x_input, (hidden_gate, cell_gate))
                )
        return output_gate, (hidden_gate, cell_gate)
    
    def forward_first_layer(self, input):
        x, (hidden_gate, cell_gate) = input
        batch_size, seq_size, _, h, w = x.size()
        all_gates = torch.zeros(3, batch_size, seq_size, self.hidden_size, h, w).to(x.device)
        for idx in range(seq_size):
            x_input = x[:, idx, :, :, :]
            output_gate, (hidden_gate, cell_gate) = self.forward_cell(
                (x_input, (hidden_gate, cell_gate))
                )
            self.add(idx, (output_gate, hidden_gate, cell_gate), all_gates)
        return all_gates[0], (all_gates[1], all_gates[2])
    
    def forward_intermediate_layer(self, input):
        x, (hidden_gates, cell_gates) = input
        batch_size, seq_size, _, h, w = x.size()
        all_gates = torch.zeros(3, batch_size, seq_size, self.hidden_size, h, w).to(x.device)
        for idx in range(seq_size):
            x_input = x[:, idx, :, :, :]
            output_gate, (hidden_gate, cell_gate) = self.forward_cell(
                (x_input, (hidden_gates[:, idx, :, :], cell_gates[:, idx, :, :]))
                )
            self.add(idx, (output_gate, hidden_gate, cell_gate), all_gates)
        return all_gates[0], (all_gates[1], all_gates[2])

    def forward_last_layer(self, input):
        x, (hidden_gates, cell_gates) = input
        _, seq_size, _, _, _ = x.size()
        for idx in range(seq_size):
            x_input = x[:, idx, :, :, :]
            output_gate, (hidden_gate, cell_gate) = self.forward_cell(
                (x_input, (hidden_gates[:, idx, :, :], cell_gates[:, idx, :, :]))
                )
        return output_gate, (hidden_gate, cell_gate)

    def forward_cell(self, input):
        x, (hidden_gate, cell_gate) = input
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
    
    def add(self, idx, gates, all_gates):
        all_gates[0, :, idx, :, :, :] = gates[0]
        all_gates[1, :, idx, :, :, :] = gates[1]
        all_gates[2, :, idx, :, :, :] = gates[2]


class ConvLSTM(nn.Module):
    def __init__(self, input_size: Union[List, Tuple], hidden_size: Union[int, List, Tuple], 
                 kernel_size: Optional[Union[int, List, Tuple]] = 3, num_layers: Optional[int] = 1
                 ):
        super().__init__()
        if type(kernel_size) not in [int, list, tuple]:
            raise TypeError("type of kernel_size is not any of int, list, or tuple".format(type(kernel_size)))
        if type(hidden_size) not in [int, list, tuple]:
            raise TypeError("type of hidden_size is not any of int, list, or tuple".format(type(hidden_size)))
        if type(kernel_size) is int:
            kernel_size = [kernel_size] * num_layers
        if type(hidden_size) is int:
            hidden_size = [hidden_size] * num_layers
        if type(kernel_size) in (list, tuple) and len(kernel_size) != num_layers:
            raise Exception("There are {} kernel_size parameters that is not equivalent to num_layers {}".format(
                len(kernel_size), num_layers))
        if type(hidden_size) in (list, tuple) and len(hidden_size) != num_layers:
            raise Exception("There are {} hidden_size parameters that is not equivalent to num_layers {}".format(
                len(hidden_size), num_layers))
        
        if num_layers == 1:
            layers = [(True, True)]
        else:
            layers = [(False, False)] * num_layers
            layers[0] = (True, False)
            layers[-1] = (False, True)
        
        input_size = [input_size] + [(hidden_size[0], *input_size[1:])] * (num_layers - 1)
        self.cell = nn.Sequential(*[
            ConvLSTMCell(
                input_size=input_size[i], kernel_size=kernel_size[i], 
                hidden_size=hidden_size[i], first_layer=layers[i][0],
                last_layer=layers[i][1]
                ) 
                for i in range(num_layers)
                ]
                )
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size, seq_size, _, w, h = x.size()
        hidden_gate = torch.zeros(batch_size, self.hidden_size[0], w, h).to(torch.float32).to(x.device)
        cell_gate = torch.zeros(batch_size, self.hidden_size[0], w, h).to(torch.float32).to(x.device)
        output_gate, (hidden_gate, cell_gate) = self.cell(
            (x, (hidden_gate, cell_gate))
            )
        return output_gate, (hidden_gate, cell_gate)
   
if __name__ == "__main__":
    batch_size = 16
    hidden_size = 32
    kernel_size = 3
    input_size = (3, 256, 256)

    x = torch.empty(size=(batch_size, 2, *input_size))
    model = ConvLSTM(input_size=input_size, hidden_size=hidden_size, kernel_size=kernel_size, num_layers=1)
    o, (h, c) = model(x)
    print(o.shape, h.shape, c.shape)
    