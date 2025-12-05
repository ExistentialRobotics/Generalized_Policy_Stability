#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import torch

# class LyapunovNet(torch.nn.Module):
#     def __init__(self, n_input, n_hidden, n_output):
#         super(LyapunovNet, self).__init__()
#         torch.manual_seed(2)
#         self.layer1 = torch.nn.Linear(n_input, n_hidden)
#         self.layer2 = torch.nn.Linear(n_hidden, n_hidden)
#         self.layer3 = torch.nn.Linear(n_hidden, n_hidden)
#         self.layer4 = torch.nn.Linear(n_hidden, n_output)

#         self.activation = torch.nn.Tanh()
#         torch.nn.init.xavier_uniform_(self.layer1.weight)
#         torch.nn.init.xavier_uniform_(self.layer2.weight)
#         torch.nn.init.xavier_uniform_(self.layer3.weight)
#         torch.nn.init.xavier_uniform_(self.layer4.weight)

#     def forward(self, x):
#         h_1 = self.activation(self.layer1(x))
#         h_2 = self.activation(self.layer2(h_1))
#         h_3 = self.activation(self.layer3(h_2))
#         out = self.layer4(h_3)
#         return out


class LyapunovNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_layers=3, leaky_relu_slope=0.01):
        super(LyapunovNet, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(n_input, n_hidden))

        # Additional hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(torch.nn.Linear(n_hidden, n_hidden))

        self.output_layer = torch.nn.Linear(n_hidden, 1)  # Always output 1 dimension
        self.activation = torch.nn.LeakyReLU(negative_slope=leaky_relu_slope)
        # self.activation = torch.nn.Tanh()
        # Initialize weights using Kaiming initialization
        for layer in self.layers:
            torch.nn.init.kaiming_normal_(
                layer.weight, a=leaky_relu_slope, nonlinearity="leaky_relu"
            )
            torch.nn.init.zeros_(layer.bias)

        torch.nn.init.kaiming_normal_(
            self.output_layer.weight, a=leaky_relu_slope, nonlinearity="leaky_relu"
        )
        torch.nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        out = self.output_layer(x)
        return out  # This will always have shape [batch_size, 1]
