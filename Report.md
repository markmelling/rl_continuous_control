
# Reinforcement Learning - Continuous Control

### Introduction

This project uses the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment to demonstrate ML agents solving an environment with a coninuous action space.


### Solving the Environment

This project uses the single variant of the Unity environment and demonstrates the solution using two differen algorithms:
- Deep Deterministic Policy Gradient (DDPG) (https://arxiv.org/abs/1509.02971v6)
- Twin Delayed Deep Deterministic (TD3) (https://arxiv.org/abs/1802.09477v3)

Both algorithms were successfully able to achieve an average reward of > 30 for 100 episodes.


### Implementation
Where are the files
What is in the files

where are the models

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

#### Plot of rewards

As can be seen from the chart  


### Twin Delayed Deep Deterministic 
TD3 builds on DDPG, like DDPG it is an model-free, off-policy algorithm that supports continuous action spaces. DDPG can over estimate Q-values which leads to the policy breaking, to tackle this TD3 introduces three improvements:
- TD3 uses 2 Q-learning networks and in calculating the Bellman Optimality Equation it takes the minimium of these two networks (target = rewards + (gamma * (1 - dones) * torch.min(q_1, q_2)))
- The policy (and target network) are updated less frequently (I followed the recommended one policy update for two Q function updates)
- Adds noise to the target action to make it harder to exploit Q-function errors

The TD3 algorithm significantly reduced the time to 'solve' environment. The TD3 was a lot more stable and took about 90,000 steps to reach a score of 30+ wheras DDPG was a lot less stable and took over 800,000 steps to reliably achieve a score of over 30.

#### Plot of rewards

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

### Future work
Other future work considered:
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
