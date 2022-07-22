"""
"""
import random
import copy
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim

import config
from network.actor import Actor
from network.critic import Critic
torch.manual_seed(config.SEED)
random.seed(config.SEED)

device = torch.device(config.DEVICE)

class DDPG():        
    def __init__(self, input_dims, action_dims, num_agents):        
        self.input_dims = input_dims
        self.action_dims = action_dims
        self.num_agents = num_agents

        self.actor_local = Actor(self.input_dims, self.action_dims).to(device)
        self.actor_target = Actor(self.input_dims, self.action_dims).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.LR)
        self.critic_local = Critic(self.input_dims, self.action_dims).to(device)
        self.critic_target = Critic(self.input_dims, self.action_dims).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.LR)

        self.noise = OUNoise((self.num_agents, self.action_dims))
        self.memory = ReplayBuffer(self.input_dims, self.action_dims)
    
    def step(self, state, action, reward, next_state, done):
        for i in range(self.num_agents):
            self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])
        batch = self.memory.sample()
        if batch is None:
            return
        self.learn(batch)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()        
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, batch):
        states, actions, rewards, next_states, dones = batch

        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (config.GAMMA * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_model(self.critic_local, self.critic_target)
        self.update_model(self.actor_local, self.actor_target)

    def update_model(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(config.TAU*local_param.data + (1.0-config.TAU)*target_param.data)

class OUNoise:
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.size = size        
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma        
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state        
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    def __init__(self, input_dim, action_dims, mem_size=config.BUFFER_SIZE):
        self.mem_size = mem_size
        self.mem_ctr = 0
        self.state = np.zeros((mem_size, input_dim), dtype=np.float32)
        self.action = np.zeros((mem_size, action_dims), dtype=np.int32)
        self.next_state = np.zeros((mem_size, input_dim), dtype=np.float32)
        self.reward = np.zeros((mem_size, 1), dtype=np.float32)
        self.done = np.zeros((mem_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        index = self.mem_ctr % self.mem_size
        self.state[index] = state
        self.action[index] = action
        self.next_state[index] = next_state
        self.reward[index] = reward
        self.done[index] = done
        self.mem_ctr += 1

    def sample(self):
        if self.mem_ctr < config.BATCH_SIZE:
            return None
        max_sample = min(self.mem_size, self.mem_ctr)
        batch = np.random.choice(max_sample, config.BATCH_SIZE, replace=False)
        state = torch.from_numpy(self.state[batch]).float().to(device)
        action = torch.from_numpy(self.action[batch]).float().to(device)
        reward = torch.from_numpy(self.reward[batch]).float().to(device)
        next_state = torch.from_numpy(self.next_state[batch]).float().to(device)
        done = torch.from_numpy(self.done[batch].astype(np.uint8)).float().to(device)
        return state, action, reward, next_state, done
