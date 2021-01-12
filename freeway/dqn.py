#!/usr/bin/env python
# -*- coding: utf-8 -*
import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    利用Pytorch实现论文中的提出的DQN算法
    """
    def __init__(self, in_channels, num_actions):
        super().__init__()
        network = [
            torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        ]

        self.network = nn.Sequential(*network)

    def forward(self, x):
        actions = self.network(x)
        return actions