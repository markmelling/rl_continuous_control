import numpy as np
import random
import copy
from collections import namedtuple, deque

from lib.model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

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

class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return

class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)
        self.coef = coef

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return self.coef * x

class DDPG_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, action_low=-1.0, action_high=1.0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.action_low = action_low
        self.action_high = action_high
        self.total_steps = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Netwo rk (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
     

        # NOISE PROCESS
        # self.noise = OUNoise(action_size, random_seed)

        self.noise = OrnsteinUhlenbeckProcess(
            size=(self.action_size,), std=LinearSchedule(0.2))

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # MM Normalizers 
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
    
    # Add step to memory and learn from experiences 
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.total_steps += 1
        state = self.state_normalizer(state)
        next_state = self.state_normalizer(next_state)
        reward = self.reward_normalizer(reward)
        action = np.clip(action, self.action_low, self.action_high)
        self.memory.add(state, action, reward, next_state, done)

        if done[0]:
            self.noise.reset()
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            # MM I've added in - learn every n steps
            if self.total_steps % LEARN_EVERY_STEPS == 0:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    # given a state what should be the action?
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        # eval
        # equivalent to model.train(False)
        # BatchNorm layers use running statistics
        # dropout layers are deactivated
        self.actor_local.eval()
        # MM in 'the other one' actions are only taken from
        # the NN after a 'warm_up' number of steps
        # before this actions are just sampled
        # could use np.random.uniform(low=-1.0, high=1.0, size=(1,4))
        # to achieve this

        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, self.action_low, self.action_high)

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
        actions_next = self.actor_target(next_states) # (1)


        Q_targets_next = self.critic_target(next_states, actions_next) # (2)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) # (3)
        # Compute critic loss
        Q_expected = self.critic_local(states, actions) # (4)
        # critic_loss = F.mse_loss(Q_expected, Q_targets) # (5) - TODO compare with (5)
        critic_loss = (Q_expected - Q_targets).pow(2).mul(0.5).sum(-1).mean() # (5)
        # Minimize the loss
        self.critic_optimizer.zero_grad() # (6)
        critic_loss.backward() # (7)
        # MM Looks like I added this in
        # torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1) # (8) TODO What does it do? should this be removed? 
        self.critic_optimizer.step() # (9)

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states) # (10)
        # TODO no states.detach() in (11) 
        # actor_loss = -self.critic_local(states, actions_pred).mean() # (11)
        actor_loss = -self.critic_local(states.detach(), actions_pred).mean() # (11)
        # Minimize the loss
        self.actor_optimizer.zero_grad() # (12)
        actor_loss.backward() # (13)
        self.actor_optimizer.step() # (14)

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU) # (15)
        self.soft_update(self.actor_local, self.actor_target, TAU) # (15)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self):
        torch.save(self.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(self.critic_local.state_dict(), 'checkpoint_critic.pth')

    def load(self):
        self.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
        self.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val

class OrnsteinUhlenbeckProcess():
    def __init__(self, size, std, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std() * np.sqrt(
            self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
