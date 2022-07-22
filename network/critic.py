"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, input_dims, action_dims, fc1_dims=256, fc2_dims=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims + action_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)

    def forward(self, state, action):
        state_value = F.relu(self.fc1(state))
        state_action_value = torch.cat((state_value, action), dim=1)
        state_action_value = F.relu(self.fc2(state_action_value))
        return self.fc3(state_action_value)