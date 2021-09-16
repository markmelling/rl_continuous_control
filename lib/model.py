import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import *

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

# Although not the same as FCBody in https://github.com/ShangtongZhang/DeepRL
# The idea of separating this part of the NN came from this repo
class FC_Core(nn.Module):
    def __init__(self, state_size, hidden_units=(64, 64), activation_fn=F.relu):
        super(FC_Core, self).__init__()
        dims = (state_size,) + hidden_units

        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.activation_fn = activation_fn
        self.output_size = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        return x


# Although not the same as in https://github.com/ShangtongZhang/DeepRL
# The idea of creating a single 'ActorCritic' network came from this repo
class Deterministic_ActorCritic_Net(nn.Module):
    def __init__(self,
                 state_size,
                 action_size):
        super(Deterministic_ActorCritic_Net, self).__init__()
        self.actor_body = FC_Core(state_size, (400, 300), activation_fn=F.relu)
        self.critic_body = FC_Core(state_size + action_size, (400, 300), activation_fn=F.relu)
        self.fc_actor = layer_init(nn.Linear(self.actor_body.output_size, action_size), 1e-3)
        self.fc_critic = layer_init(nn.Linear(self.critic_body.output_size, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_actor.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_params, lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=1e-3)
        self.to(device)

    def forward(self, obs):
        obs = tensor(obs)
        action = self.actor(obs)
        return action

    def actor(self, states):
        return torch.tanh(self.fc_actor(self.actor_body(states)))

    def critic(self, states, actions):
        return self.fc_critic(self.critic_body(torch.cat([states, actions], dim=1)))


class TD3_Net(nn.Module):
    def __init__(self,
                 state_size,
                 action_size,
                 ):
        super(TD3_Net, self).__init__()
        self.actor_body = FC_Core(state_size, (400, 300), activation_fn=F.relu)
        self.critic_body_1 = FC_Core(state_size + action_size, (400, 300), activation_fn=F.relu)
        self.critic_body_2 = FC_Core(state_size + action_size, (400, 300), activation_fn=F.relu)

        self.fc_actor = layer_init(nn.Linear(self.actor_body.output_size, action_size), 1e-3)
        self.fc_critic_1 = layer_init(nn.Linear(self.critic_body_1.output_size, 1), 1e-3)
        self.fc_critic_2 = layer_init(nn.Linear(self.critic_body_2.output_size, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_actor.parameters())
        self.critic_params = list(self.critic_body_1.parameters()) + list(self.fc_critic_1.parameters()) +\
                             list(self.critic_body_2.parameters()) + list(self.fc_critic_2.parameters())

        self.actor_optimizer = torch.optim.Adam(self.actor_params, lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=1e-3)

        self.to(device)

    def forward(self, obs):
        obs = tensor(obs)
        return torch.tanh(self.fc_actor(self.actor_body(obs)))

    def q(self, obs, a):
        obs = tensor(obs)
        a = tensor(a)
        x = torch.cat([obs, a], dim=1)
        q_1 = self.fc_critic_1(self.critic_body_1(x))
        q_2 = self.fc_critic_2(self.critic_body_2(x))
        return q_1, q_2

