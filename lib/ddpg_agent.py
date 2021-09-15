import numpy as np
import random
import copy
from collections import namedtuple, deque

from lib.model import Deterministic_ActorCritic_Net, Critic

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from lib.model import Deterministic_ActorCritic_Net
from lib.replay_buffer import ReplayBuffer
from lib.utils import *
from lib.base_agent import BaseAgent

# BUFFER_SIZE = int(1e5)  # replay buffer size
BUFFER_SIZE = int(1e6)  # MM replay buffer size
BATCH_SIZE = 128        # minibatch size - MM is 100
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

class DDPG_Agent(BaseAgent):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                 name,
                 state_size,
                 action_size,
                 random_seed,
                 action_low=-1.0,
                 action_high=1.0,
                 warm_up=int(1e4)):
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
        self.total_steps = 0

        def create_nn():
            return Deterministic_ActorCritic_Net(state_size, action_size)

        self.local_network = create_nn()
        self.target_network = create_nn()
        self.target_network.load_state_dict(self.local_network.state_dict())

        # NOISE PROCESS
        self.noise = OrnsteinUhlenbeckProcess(
            size=(self.action_size,), std=LinearSchedule(0.2))

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

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.target_network.actor(next_states) # (1)
        Q_targets_next = self.target_network.critic(next_states, actions_next) # (2)

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) 
        # Compute critic loss
        Q_expected = self.local_network.critic(states, actions)
        # critic_loss = F.mse_loss(Q_expected, Q_targets) # (5)
        critic_loss = (Q_expected - Q_targets).pow(2).mul(0.5).sum(-1).mean() 
        # Minimize the loss
        self.local_network.zero_grad()
        critic_loss.backward()
        self.local_network.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.local_network.actor(states)
        actor_loss = -self.local_network.critic(states.detach(), actions_pred).mean()
        # Minimize the loss
        self.local_network.zero_grad()
        actor_loss.backward()
        self.local_network.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.target_network, self.local_network, TAU) 

    def soft_update(self, target, src, tau):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - tau) +
                               param * tau)


