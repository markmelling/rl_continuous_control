[image1]: https://raw.githubusercontent.com/markmelling/rl_continuous_control/main/ddpg_learning_rate.png
[image2]: https://raw.githubusercontent.com/markmelling/rl_continuous_control/main/td3_learning_rate.png 
[image3]: https://raw.githubusercontent.com/markmelling/rl_continuous_control/main/ddpg_test_scores.png
[image4]: https://raw.githubusercontent.com/markmelling/rl_continuous_control/main/td3_test_scores.png

# Reinforcement Learning - Continuous Control

### Introduction

This project uses the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment to demonstrate ML agents solving an environment with a coninuous action space.


### Solving the Environment

This project uses the single variant of the Unity environment and demonstrates the solution using two differen algorithms:
- Deep Deterministic Policy Gradient (DDPG) (https://arxiv.org/abs/1509.02971v6)
- Twin Delayed Deep Deterministic (TD3) (https://arxiv.org/abs/1802.09477v3)

Both algorithms were successfully able to achieve an average reward of > 30 for 100 episodes.


### Implementation

The code to train or test either of the implemented models can either be run from the command line or from a Jupyter notebook (see `Continuous_Control.ipynb`)

run_agent.py in the root of this repo is used to test or evaluate the models.

All other source files are in the lib folder

- environments.py - provides a wrapper around a unity environment
- ddpg_agent.py - DDPG_Agent class implements the DDPG algorithm
- td3_agent.py - TD3_Agent class implements the DDPG algorithm
- model.py 
  - Implementations of a Deterministic Actor Critic Neural Network 
  - Implementations of a Neural Network supporting TD3
- replay_buffer.py - experience replay buffer
- utils - various useful functions and noise classes


### Deep Deterministic Policy Gradient
Deep Deterministic Policy Gradient is a model-free, off-policy algorithm for learning continuous actions. It combines Deterministic Policy Gradient and Deep Q-Network. 

DDPG uses an experience replay buffer along with a target network, to stabilize learning.

DDPG uses an Actor-Critic algorithm where a value based function and policy based function are merged.
The basic idea is to split the model in two: one for computing an action based on a state and another one to produce the Q values of the action.


- the actor inputs the state and determines the best action
- the critic evaluates the state and action (from the actor) by computing the Q-value 
     
Gradient ascent (not descent) is used to maximise the Q-value and update the weights.

For DDPG it also uses a target Actor-Critic network to add stability to the training. The target network's weights are gradually updated from the network.

An experience replay buffer is used to learn from previous experiences. These samples are randomly sampled and used to learn from.

An Ornstein-Uhlenbeck process is used for generating noise to implement better exploration by the Actor network.


#### Hyper-pararmeters

- Replay buffer size: 1e6
- Replay batch (sample) size:  128 
- Discount factor (gamma): 0.99
- Soft update rate (tau): 5e-3
- layer initialization: orthogonal,  weight scale: 1e-3 
- Optimizer: Adam, learning_rate: 1e-3

#### Neural network architecture
- actor hidden units = 400, 300
- Actor learning rate: 1e-3
- critic hidden units = 400, 300
- Critic learning rate: 1e-3

#### Plot of rewards during training
![Learning rate][image1]


#### Distribution of episode rewards over 100 test episodes
![test scores][image3]

#### Model weights
The model weights for a DDPG agent that produces scores of 30+ is stored in `Reacher_DDPG_Trained.pth`


### Twin Delayed Deep Deterministic 
TD3 builds on DDPG, like DDPG it is an model-free, off-policy algorithm that supports continuous action spaces. DDPG can over estimate Q-values which leads to the policy breaking, to tackle this TD3 introduces three improvements:
- TD3 uses 2 Q-learning networks and in calculating the Bellman Optimality Equation it takes the minimium of these two networks (target = rewards + (gamma * (1 - dones) * torch.min(q_1, q_2)))
- The policy (and target network) are updated less frequently (I followed the recommended one policy update for two Q function updates)
- Adds noise to the target action to make it harder to exploit Q-function errors

A gaussian process is used for generating noise.

#### Hyper-pararmeters

- Replay buffer size: 1e6
- Replay batch (sample) size:  128 
- Discount factor (gamma): 0.99
- Soft update rate (tau): 5e-3
- layer initialization: orthogonal,  weight scale: 1e-3 
- Optimizer: Adam, learning_rate: 1e-3

- noise: 0.2,
- noise clipping: 0.5
- delay in updating the policy and target network: 2

#### Neural network architecture
- actor hidden units = 400, 300
- Actor learning rate: 1e-3
- critic hidden units = 400, 300
- Critic learning rate: 1e-3

#### Plot of rewards during training
![Learning rate][image2]

#### Distribution of episode rewards over 100 test episodes
![test scores][image4]

#### Model weights
The model weights for a DDPG agent that produces scores of 30+ is stored in `Reacher_TD3_Trained.pth`


### Comparison of DDPG and TD3
The TD3 algorithm significantly reduced the time to 'solve' the environment. TD3 was a lot more stable and took about 90,000 steps to reach a score of 30+ wheras with DDPG there was a lot more variation in the scores during training and it took over 800,000 steps to reliably achieve a score of over 30.


### Future work
The length of time that it takes to train a model is considerable on my current setup and really is a barrier to testing and experimenting. I need to investigate both improved local versions (faster computer) and 'in the cloud' options, both in terms of the reduction in time for training to take and cost.

Other future work worth considering:
#### Multi-agents and additional algorithms
Add support for multiple agents and implement some of the other well know algorithms and compare their performance (e.g. PPO and A2C)

#### Replay buffer
Experiment with prioritised experience replay buffer.


### Glossary 

#### Model-free
A model-free algorithm does not use a model of the environment. That is it doesn't use a function which predicts state transitions or rewards.
Q-learning is an example of a model-free algorithm.
#### Off-policy (from Richard Sutton's book)
In Q-learning, the agent learns an optimal policy with the help of a greedy policy and behaves using policies of other agents. Q-learning is called off-policy because the updated policy is different from the behaviour policy. In other words, it estimates the reward for future actions and appends a value to the new state without actually following any greedy policy.
#### Experience Replay
As experiences (state, action, reward, next state) with the environment happen they are stored and then subsequently sampled to learn from.
