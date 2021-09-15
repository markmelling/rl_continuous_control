import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from lib.model import TD3_Net
from lib.replay_buffer import ReplayBuffer
from lib.utils import *
from lib.base_agent import BaseAgent

# BUFFER_SIZE = int(1e5)  # replay buffer size
BUFFER_SIZE = int(1e6)  # MM replay buffer size
BATCH_SIZE = 128        # minibatch size - MM is 100
# BATCH_SIZE = 100
GAMMA = 0.99            # discount factor - MM is same 
# TAU = 1e-3              # for soft update of target parameters
TAU = 5e-3              # MM Changed to match 

# LR_ACTOR = 1e-4         # learning rate of the actor 
LR_ACTOR = 1e-3         # MM changed learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# LEARN_EVERY_STEPS = 20
LEARN_EVERY_STEPS = 1


class TD3_Agent(BaseAgent):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                 name,
                 state_size,
                 action_size,
                 random_seed,
                 action_low=-1.0,
                 action_high=1.0,
                 warm_up=int(1e4),
                 td3_noise=0.2,
                 td3_noise_clip=0.5,
                 td3_delay=2,
                 ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.action_low = action_low
        self.action_high = action_high
        self.warm_up = warm_up
        self.td3_noise = td3_noise
        self.td3_noise_clip = td3_noise_clip
        self.td3_delay = td3_delay
        self.total_steps = 0

        def create_nn():
            return TD3_Net(state_size, action_size)

        self.local_network = create_nn()
        self.target_network = create_nn()
        self.target_network.load_state_dict(self.local_network.state_dict())

        # NOISE PROCESS
        self.noise = GaussianProcess(
            size=(action_size,), std=LinearSchedule(0.1))

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    # given a state what should be the action?
    def act(self, state, train=True):
        """Returns actions for given state as per current policy."""
        if train and self.total_steps < self.warm_up:
            action = np.random.uniform(low=self.action_low,
                                       high=self.action_high,
                                       size=(1,self.action_size))
        else:
            action = self.local_network(state)
            action = to_np(action)
            if train:
                action += self.noise.sample()
        return np.clip(action, self.action_low, self.action_high)

    # Add step to memory and learn from experiences 
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.total_steps += 1

        self.memory.add(state, action, reward, next_state, done)

        if done[0]:
            self.noise.reset()
        # Learn, if enough samples are available in memory
        if self.memory.size() > BATCH_SIZE:
            if self.total_steps % LEARN_EVERY_STEPS == 0:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        actions_next = self.target_network(next_states) 
        noise = torch.randn_like(actions_next).mul(self.td3_noise)
        noise = noise.clamp(-self.td3_noise_clip, self.td3_noise_clip)

        actions_next = (actions_next + noise).clamp(self.action_low, self.action_high)

        q_1, q_2 = self.target_network.q(next_states, actions_next)
        target = rewards + (gamma * (1 - dones) * torch.min(q_1, q_2))
        target = target.detach()

        q_1, q_2 = self.local_network.q(states, actions)
        critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)

        # Minimize the loss
        self.local_network.zero_grad()
        critic_loss.backward()
        self.local_network.critic_optimizer.step()

        self.soft_update(self.target_network, self.local_network, TAU)

        if self.total_steps % self.td3_delay:
            action = self.local_network(states)
            policy_loss = -self.local_network.q(states, action)[0].mean()

            self.local_network.zero_grad()
            policy_loss.backward()
            self.local_network.actor_optimizer.step()

            self.soft_update(self.target_network, self.local_network, TAU)

    def soft_update(self, target, src, tau):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - tau) +
                               param * tau)

