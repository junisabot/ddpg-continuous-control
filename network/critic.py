"""
"""
import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(config.SEED)

class Critic(nn.Module):
    def __init__(self, input_dims, action_dims, fc1_dims=512, fc2_dims=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims + action_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)
        self._init_weight()

    def forward(self, state, action):
        state_value = F.relu(self.fc1(state))
        state_action_value = torch.cat((state_value, action), dim=1)
        state_action_value = F.relu(self.fc2(state_action_value))
        return self.fc3(state_action_value)

    def _hidden_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)

    def _init_weight(self):
        self.fc1.weight.data.uniform_(*self._hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self._hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)