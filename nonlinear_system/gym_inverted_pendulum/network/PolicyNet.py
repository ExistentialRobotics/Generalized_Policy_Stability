#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import torch


class PolicyNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=3, leaky_relu_slope=0.01):
        super(PolicyNet, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(n_input, n_hidden))

        # Additional hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(torch.nn.Linear(n_hidden, n_hidden))

        self.output_layer = torch.nn.Linear(n_hidden, n_output)
        self.activation = torch.nn.LeakyReLU(negative_slope=leaky_relu_slope)

        # Initialize weights using Kaiming initialization
        for layer in self.layers:
            torch.nn.init.kaiming_normal_(
                layer.weight, a=leaky_relu_slope, nonlinearity="leaky_relu"
            )
            torch.nn.init.zeros_(layer.bias)

        # Initialize output layer with smaller weights
        torch.nn.init.kaiming_normal_(
            self.output_layer.weight, a=leaky_relu_slope, nonlinearity="leaky_relu"
        )
        torch.nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        # No activation applied to the output layer
        out = self.output_layer(x)
        return out


class StepNet(torch.nn.Module):
    def __init__(self, n_input=3, n_hidden=32, n_steps=40, n_layers=2):
        super(StepNet, self).__init__()

        self.n_steps = n_steps

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(n_input, n_hidden))

        # Additional hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(torch.nn.Linear(n_hidden, n_hidden))

        # Output raw logits
        self.output_layer = torch.nn.Linear(n_hidden, n_steps)
        self.activation = torch.nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        for layer in self.layers:
            x = self.activation(layer(x))

        # Get raw logits
        logits = self.output_layer(x)

        # Apply softmax and scale by M to ensure sum equals M
        weights = self.n_steps * torch.nn.functional.softmax(logits, dim=1)

        return weights
