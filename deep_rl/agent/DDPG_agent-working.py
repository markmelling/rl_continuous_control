#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
# import torchvision


class DDPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        # self.task = config.task_fn()
        # config.eval_env = self.task
        # self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.network = config.network_fn()

        # self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
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
            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next) # (1)
            q_next = self.target_network.critic(phi_next, a_next) # (2)
            q_next = config.discount * mask * q_next # (3)
            q_next.add_(rewards) # (3)
            q_next = q_next.detach()
            phi = self.network.feature(states) # (4)
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
            phi = self.network.feature(states)
            action = self.network.actor(phi) # (10)
            policy_loss = -self.network.critic(phi.detach(), action).mean() # (11)

            self.network.zero_grad() # (12)
            policy_loss.backward() # (13)
            self.network.actor_opt.step() # (14)

            self.soft_update(self.target_network, self.network) # (15)
