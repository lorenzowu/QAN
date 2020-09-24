import torch
import torch.nn as nn
import math
from torch.nn import init


class ConvLSTMCell_LayerNorm(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride,
                 bias=True, layer_norm=True, elementwise_affine=False, reset=False):
        # input_size, hidden_size --> C,H,W
        # kernel_size, stride --> int
        super(ConvLSTMCell_LayerNorm, self).__init__()
        input_dim, input_height, input_width = input_size
        hidden_dim, hidden_height, hidden_width = hidden_size
        padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        if layer_norm is True:
            self.LN = nn.LayerNorm(normalized_shape=hidden_size, elementwise_affine=elementwise_affine)

        self.layer_norm = layer_norm
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_dim

        if reset is True:
            self.reset_parameters()

    def forward(self, input_tensor, cur_state=None):
        # batch_size, channel, height, width
        if cur_state is None:
            h_cur = input_tensor.new_zeros(input_tensor.shape, requires_grad=False)
            c_cur = h_cur
        else:
            h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        if self.layer_norm is True:
            cc_i = self.LN(cc_i)
            cc_f = self.LN(cc_f)
            cc_o = self.LN(cc_o)
            cc_g = self.LN(cc_g)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g

        if self.layer_norm is True:
            c_next = self.LN(c_next)

        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size[0] * self.hidden_size[1] * self.hidden_size[2])
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)


class ConvLSTM_LayerNorm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, kernel_size=3, stride=1,
                 bias=True, layer_norm=True, elementwise_affine=False, reset=False):
        super(ConvLSTM_LayerNorm, self).__init__()

        cell_list = []
        for i in range(0, num_layers):
            cell_list.append(ConvLSTMCell_LayerNorm(input_size=input_size,
                                                    hidden_size=hidden_size,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    bias=bias,
                                                    layer_norm=layer_norm,
                                                    elementwise_affine=elementwise_affine,
                                                    reset=reset))
        self.cell_list = nn.ModuleList(cell_list)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, input_tensor, hidden_state=None):
        # batch_size, seq_len, input_channel, input_height, input_width

        if hidden_state is None:
            h_cur = input_tensor.new_zeros(
                (input_tensor.size(0), input_tensor.size(2), input_tensor.size(3), input_tensor.size(4)),
                requires_grad=False)
            c_cur = h_cur

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            output_inner = []
            h = h_cur
            c = c_cur
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=(h, c))
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

        return cur_layer_input
