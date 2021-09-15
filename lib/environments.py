
from unityagents import UnityEnvironment
import numpy as np


class UnityEnv():
    # TODO not sure that action_low and high are needed
    def __init__(self, name, path, train_mode, action_low=-1.0, action_high=1.0):
        self.name = name
        self.path = path
        self.train_mode = train_mode
        self.action_low = action_low
        self.action_high = action_high

        self.env = UnityEnvironment(file_name=self.path)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        print('brain_name:', self.brain_name)
        # reset the environment
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]

        # number of agents
        self.num_agents = len(env_info.agents)
        print('Number of agents:', self.num_agents)

        # size of each action
        self.action_size = self.brain.vector_action_space_size
        print('Size of each action:', self.action_size)

        # examine the state space 
        self.states = env_info.vector_observations
        self.state_size = self.states.shape[1]
        print('There are {} agents. Each observes a state with length: {}'.format(self.states.shape[0], self.state_size))
        print('The state for the first agent looks like:', self.states[0])


    def reset(self, train_mode=None):
        if train_mode == None:
            train_mode = self.train_mode
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.total_rewards = 0
        return env_info.vector_observations

    def close(self):
        if self.env:
            self.env.close()

    def step(self, actions):
        actions = np.clip(actions, self.action_low, self.action_high)
        env_info = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        self.total_rewards += rewards[0]
        dones = env_info.local_done                        # see if episode finished
        states = next_states                               # roll over states to next time step
        if np.any(dones):
            episodic_return = self.total_rewards
        else:
            episodic_return = None

        info = ({
          'episodic_return': episodic_return
        },)
        return next_states, rewards, dones, info
