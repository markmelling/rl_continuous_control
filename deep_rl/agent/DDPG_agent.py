#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
# import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def layer_init_mm(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class FCBody_MM(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody_MM, self).__init__()
        dims = (state_dim,) + hidden_units

        self.layers = nn.ModuleList(
            [layer_init_mm(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = gate
        self.feature_dim = dims[-1]

    def reset_noise(self):
        pass
        # if self.noisy_linear:
        #     for layer in self.layers:
        #         layer.reset_noise()

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class DummyBody_MM(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody_MM, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x

class DeterministicActorCriticNet_MM(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticNet_MM, self).__init__()
        # if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody_MM(state_dim)
        if critic_body is None: critic_body = DummyBody_MM(state_dim)
        # self.phi_body = phi_body
        self.actor_body = FCBody_MM(state_dim, (400, 300), gate=F.relu)
        self.critic_body = FCBody_MM(state_dim + action_dim, (400, 300), gate=F.relu)
        self.fc_actor = layer_init_mm(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init_mm(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_actor.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        # self.phi_params = list(self.phi_body.parameters())
        # self.actor_opt = torch.optim.Adam(self.actor_params + self.phi_params, lr=1e-3)
        # self.critic_opt = torch.optim.Adam(self.critic_params + self.phi_params, lr=1e-3)
        self.actor_opt = torch.optim.Adam(self.actor_params, lr=1e-3)
        self.critic_opt = torch.optim.Adam(self.critic_params, lr=1e-3)
        self.to(Config.DEVICE)

    def forward(self, obs):
        # phi = self.feature(obs)
        obs = tensor(obs)
        action = self.actor(obs)
        return action

    # def feature(self, obs):
    #     obs = tensor(obs)
    #     return obs
        # return self.phi_body(obs)

    def actor(self, phi):
        return torch.tanh(self.fc_actor(self.actor_body(phi)))

    def critic(self, phi, a):
        return self.fc_critic(self.critic_body(torch.cat([phi, a], dim=1)))

class DDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        # self.task = config.task_fn()
        # config.eval_env = self.task
        # self.network = config.network_fn(self.task.state_dim, self.task.action_dim)

        def create_nn():
            return DeterministicActorCriticNet_MM(
                config.state_dim, config.action_dim,
                actor_body=FCBody_MM(config.state_dim, (400, 300), gate=F.relu),
                critic_body=FCBody_MM(config.state_dim + config.action_dim, (400, 300), gate=F.relu),
                actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
                critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

        # self.network = config.network_fn()
        self.network = create_nn()

        # self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        # self.target_network = config.network_fn()
        self.target_network = create_nn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = OrnsteinUhlenbeckProcess(
            size=(config.action_dim,), std=LinearSchedule(0.2))
        self.total_steps = 0
        self.state = None

    def save(self):
        super().save('ddpg')

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    # this is for testing
    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def act(self, state, add_noise=True):
        config = self.config
        self.state = state
        if self.state is None:
            self.random_process.reset_states()
            # self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            # action = [self.task.action_space.sample()]
            # action = self.task.action_space.sample()
            # TODO
            action = np.random.uniform(low=-1.0, high=1.0, size=(1,4))
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        # action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        action = np.clip(action, -1.0, 1.0)
        return action

    def step(self, state, action, reward, next_state, done):
        config = self.config
    # def step(self):
        # config = self.config
        # if self.state is None:
        #     self.random_process.reset_states()
        #     self.state = self.task.reset()
        #     self.state = config.state_normalizer(self.state)

        # if self.total_steps < config.warm_up:
        #     # action = [self.task.action_space.sample()]
        #     # action = self.task.action_space.sample()
        #     # TODO
        #     action = np.random.uniform(low=-1.0, high=1.0, size=(1,4))
        # else:
        #     action = self.network(self.state)
        #     action = to_np(action)
        #     action += self.random_process.sample()
        # action = np.clip(action, self.task.action_space.low, self.task.action_space.high)

        # # MM Move environment (game) on a step
        # next_state, reward, done, info = self.task.step(action)


        next_state = self.config.state_normalizer(next_state)
        # self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        # store experience
        self.replay.feed(dict(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1-np.asarray(done, dtype=np.int32),
        ))

        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.replay.size() >= config.warm_up:
            # MM do some learning
            transitions = self.replay.sample()
            # print('transitions.shape', transitions.shape)
            states = tensor(transitions.state)
            # print('states.shape', states.shape)
            actions = tensor(transitions.action)
            rewards = tensor(transitions.reward).unsqueeze(-1)
            next_states = tensor(transitions.next_state)
            mask = tensor(transitions.mask).unsqueeze(-1)

            # compute critic loss
            # phi_next = self.target_network.feature(next_states)
            phi_next = tensor(next_states)
            a_next = self.target_network.actor(phi_next) # (1)
            q_next = self.target_network.critic(phi_next, a_next) # (2)
            q_next = config.discount * mask * q_next # (3)
            q_next.add_(rewards) # (3)
            q_next = q_next.detach()
            phi = tensor(states)
            # phi = self.network.feature(states) # (4)

            # print('phi.shape', phi.shape)
            # print('actions.shape', actions.shape)
            q = self.network.critic(phi, actions) # (4)
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean() # (5)

            # update (local) critic network
            self.network.zero_grad() # (6)
            critic_loss.backward() # (7)
            # TODO no (8)
            self.network.critic_opt.step() # (9)


            # compute (local) actor loss
            phi = tensor(states)
            # phi = self.network.feature(states)
            action = self.network.actor(phi) # (10)
            policy_loss = -self.network.critic(phi.detach(), action).mean() # (11)

            self.network.zero_grad() # (12)
            policy_loss.backward() # (13)
            self.network.actor_opt.step() # (14)

            self.soft_update(self.target_network, self.network) # (15)
