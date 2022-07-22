"""
"""
import random
import copy
import numpy as np
from collections import namedtuple, deque
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
    def __init__(self, state_size, action_size, num_agents):
        random_seed = config.SEED
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents
        self.memory = ReplayBuffer(action_size, config.BUFFER_SIZE, config.BATCH_SIZE, random_seed)
        
        self.noise = OUNoise(action_size, random_seed)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.LR_CRITIC, weight_decay=0)

        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.LR_ACTOR)

    def step(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        if len(self.memory) > config.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, config.GAMMA)
        
    def act(self, states, add_noise=True):
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
            for i, state in enumerate(states):
                actions[i, :] = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_model(self.critic_local, self.critic_target, config.TAU)
        self.update_model(self.actor_local, self.actor_target, config.TAU)                     

    def update_model(self, regular_model, target_model, tau):
        for target_param, regular_param in zip(target_model.parameters(),regular_model.parameters()):
            target_param.data.copy_(tau * regular_param.data + (1.0 - tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size) 
        self.action_size = action_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)  
    
class OUNoise:
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.normal(loc=0, scale=1) for _ in range(len(x))])
        self.state = x + dx
        return self.state
    
    def reset(self):
        self.state = copy.copy(self.mu)