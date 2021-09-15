from unityagents import UnityEnvironment
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from lib.environments import UnityEnv
from lib.ddpg_agent import DDPG_Agent
from lib.td3_agent import TD3_Agent

import sys
import argparse


def eval_episode(env, agent):
    states = env.reset()
    while True:
        actions = agent.act(states, train=False)
        states, rewards, dones, info = env.step(actions)
        ret = info[0]['episodic_return']
        if ret is not None:
            break
    return ret

def eval_episodes(env, agent, num_episodes=20):
    total_rewards = []
    record = pd.DataFrame(columns=['time', 'score'])
    for i in range(num_episodes):
        t0 = time.time()
        episode_rewards = eval_episode(env, agent)
        total_rewards.append(np.sum(episode_rewards))
        print(f'Episodes: {i} average {np.mean(total_rewards)}')
        t1 = time.time()
        record = record.append(dict(time=round(t1-t0),
                                    score=round(np.sum(episode_rewards), 2)), ignore_index=True)
        record.to_csv(f'{agent.name}-evaluate.csv')
    return np.mean(total_rewards)


def train_agent(env, agent, max_steps=1e6, break_on_reward=35, save_interval=1e4, eval_interval=1e4):
    print('start training')
    states = env.reset()
    # print('state', states)
    record = pd.DataFrame(columns=['time', 'steps', 'average_score'])
    t0 = time.time()
    highest_reward = 0
    while True:
        # print('save_interval', config.save_interval, 'total_steps', agent.total_steps)
        if save_interval and not agent.total_steps % save_interval:
            agent.save()
        if eval_interval and not agent.total_steps == 0 and not agent.total_steps % eval_interval:
            print('agent.eval_episodes')
            average_reward = eval_episodes(env, agent, num_episodes=5)
            print(time.strftime("%H:%M:%S", time.localtime()),
                  f'After {agent.total_steps} steps ', 'average reward:', average_reward)
            t1 = time.time()
            record = record.append(dict(time=round(t1-t0),
                                        steps=agent.total_steps,
                                        average_score=round(average_reward, 2)), ignore_index=True)
            record.to_csv(f'{agent.name}.csv')
            if average_reward > highest_reward:
                highest_reward = average_reward
                agent.save(f'{agent.name}-best-so-far')
                if highest_reward > break_on_reward:
                    break

        if max_steps and agent.total_steps >= max_steps:
            # print('agent.close')
            agent.save()
            # env.close()
            break
        actions = agent.act(states)
        next_states, rewards, dones, info = env.step(actions)
        agent.step(states, actions, rewards, next_states, dones)
        states = next_states
    average_reward = eval_episodes(env, agent)
    if average_reward > highest_reward:
        highest_reward = average_reward
        agent.save(f'{agent.name}-best-so-far')
    print(time.strftime("%H:%M:%S", time.localtime()),
            f'After {agent.total_steps} steps ', 'average reward:', average_reward)
    t1 = time.time()
    record = record.append(dict(time=round(t1-t0),
                                steps=agent.total_steps,
                                average_score=round(average_reward, 2)), ignore_index=True)
    record.to_csv(f'{agent.name}.csv')

agents = {
    'ddpg': DDPG_Agent,
    'td3': TD3_Agent,
}

if __name__ == '__main__':
    print(len(sys.argv))
    parser = argparse.ArgumentParser(
        description='This program trains and tests RL models'
    )
    parser.add_argument('-n', '--name', metavar='name', help='Used as basis to store any output')
    parser.add_argument('-m', '--mode', metavar='mode', help='train or test', default='train')
    parser.add_argument('-f', '--filename', metavar='filename', help='filename of model')
    parser.add_argument('-a', '--agent', metavar='agent', required=True, help='agent - ddpg, td3, a2c, ppo')
    parser.add_argument('-s', '--steps', metavar='steps', help='Number of steps')


    args = parser.parse_args()

    print(args.mode)
    print(args.filename)
    print(args.agent)
    train_mode = True if args.mode == 'train' else False
    if args.agent not in agents:
        print('invalid agent, must be ddpg or td3')
        sys.exit()
    env = UnityEnv('Reacher', '../../Reacher.app', train_mode=train_mode)
    name = args.name if args.name else args.agent
    agent_fn = agents[args.agent]
    agent = agent_fn(name=name,
                     state_size=env.state_size,
                     action_size=env.action_size,
                     random_seed=2,
                     warm_up=int(1e4))
    if train_mode:
        max_steps = args.steps if args.steps else int(1e6)
        train_agent(env, agent, max_steps=max_steps)
    else:
        agent.load(filename=args.name)
        n_episodes = args.steps if args.steps else 100
        average_reward = eval_episodes(env, agent, num_episodes=n_episodes)
        print(f'average score for {n_episodes} episodes is {average_reward}')


