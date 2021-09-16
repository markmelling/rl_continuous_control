[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


# Continuous Control

### Introduction

This project uses the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment to demonstrate ML agents solving an environment with a continuous action space.

![Trained Agent][image1]

In the environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


### Solving the Environment

This project uses the single agent variant of the Unity environment and demonstrates the solution using two different algorithms:
- Deep Deterministic Policy Gradient (DDPG) (https://arxiv.org/abs/1509.02971v6)
- Twin Delayed Deep Deterministic (TD3) (https://arxiv.org/abs/1802.09477v3)

Both algorithms were successfully able to achieve an average reward of > 30 for 100 episodes.


## Getting Started

### Installing the dependencies
To run the code either from the command line or a Jupyter Notebooks you need make sure you have the Unity environment, Python and the required libraries installed.

#### Installing Python
The easiest way to install Python, or the correct version of Python, is to use conda, and the easiest way to install conda is to install miniconda.

If you don't have conda installed then follow the instructions here: [miniconda](https://docs.conda.io/en/latest/miniconda.html)

With conda installed, create and activate a conda environment which uses python 3.6 by entering the following commands in a terminal window

- For Linux or Mac
```shell
conda create --name drlnd python=3.6
source activate drlnd
```

- For Windows
```shell
conda create --name drlnd python=3.6
activate drlnd
```

#### Install the required python libraries
Clone this repository if you've not done it already, then navigate to the python subdirectory and install the dependencies by entering the following commands in a terminal window

```shell
git clone https://github.com/markmelling/rl_continuous_control
cd rl_continuous_control
pip install .
```

#### Install the pre-prepared Unity environment

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - ** One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in rl_continuous_control and unzip (or decompress) the file. 


## Instructions

You can train and evaluate the agents either from the notebook `Continuous_Control.ipynb` or from the command line (I've not tried it on Windows)

For the notebook follow the instructions there.

For the command line to train an agent you can run:
```

# for DDPG
python train.py -m train -a ddpg -n 'The_name_of_your_choosing'

# for TD3
python train.py -m train -a 'td3 -n 'The_name_of_your_choosing'
```

The best model weights are stored in a file with the name (-n) with a .pth extension. This will be updated each time the best score is improved
The results of intermidiate evaluations are stored in a csv file of the same base name with a .csv extension

To run an evaluation test from the command line:

```

# for DDPG
python train.py -m test -a ddpg -n 'The_name_of_your_choosing'

# for TD3
python train.py -m test -a 'td3 -n 'The_name_of_your_choosing'
```

This will return the average score for 100 episodes


### Pre-trained models
There are pre-trained models that produce 30+ rewards for both DDPG and TD3
- `Reacher_DDPG_Trained.pth`
- `Reacher_TD3_Trained.pth`

## Source code
Outside of the `Continuous_Control.ipynb` notebook there are the following python files:
- run_agent.py (can be either run from the command line or from a notebook)
All other source code files are in the lib folder

- environments.py - provides a wrapper around a unity environment
- ddpg_agent.py - DDPG_Agent class implements the DDPG algorithm
- td3_agent.py - TD3_Agent class implements the DDPG algorithm
- model.py 
  - Implementations of a Deterministic Actor Critic Neural Network 
  - Implementations of a Neural Network supporting TD3
- replay_buffer.py - experience replay buffer
- utils - various useful functions and noise classes
