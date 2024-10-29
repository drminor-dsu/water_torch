#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:23:32 2024

@author: drminor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from water_gym import Actions

import numpy as np
import random
import copy
from collections import deque
from matplotlib import pylab as plt


actions = Actions


# l1 = 15
# l2 = 150
# l3 = 100
# l4 = len(actions)

class DqnGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(DqnGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
             
        # Multi-layer GRU
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output layer (Q-values for each action)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state=None):
        
        # GRU layers
        x, hidden_state = self.gru(x, hidden_state)  
        
        # Output layer for Q-values
        x = self.output(x[:, -1, :])
        
        return x, hidden_state
    

class DqnLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(DqnLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        
        # Multi-layer LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output layer (Q-values for each action)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state=None):
        
        # LSTM layers
        x, hidden_state = self.lstm(x, hidden_state)  
        
        # Output layer for Q-values
        x = self.output(x[:, -1, :])
        
        return x, hidden_state
    

class DqnDNN(nn.Module):
    def __init__(self, l1, l2, l3, l4):
        super(DqnDNN, self).__init__()
        self.hidden1 = nn.Linear(l1, l2)
        self.hidden2 = nn.Linear(l2, l3)
        self.outputs = nn.Linear(l3,l4)
        
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.outputs(x)
        
        return x
        

class DqnCNN(nn.Module):
    def __init__(self, nout):
        super(DqnCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, padding=1)
        
        self.flat = nn.Flatten()
        
        self.fc1 = nn.Linear(in_features=8*32, out_features=64)
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(in_features=64, out_features=nout)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flat(self.pool(x))
        x = self.relu3(self.fc1(x))
        x = self.output(x)
        
        return x