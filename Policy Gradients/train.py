import gym
import numpy as np
import torch

from model import *
from history import *
from visualize import *

algo_name = 'VPG'
num_episodes = 1000
update_iter = 10

#Tracks the previous episodes to be used later
history = History()
#Simulation which we get data from
env = gym.make('CartPole-v1')
#Neural Network
policy = Policy(env)
#Handles improvement of network
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

past_ep_rewards = []

def demo():
    s = env.reset()
    while True:
        env.render()
        a = policy(s)
        s, _, done, _ = env.step(a)
        if done:
            break
    env.close()
demo()
def train():
    for ep in range(num_episodes):
        s = env.reset()

        e_reward = 0

        while True:
            a = policy(s)
            s2, reward, done, _ = env.step(a)
            e_reward += reward
            history.store(s,a,reward,done)
            if done:
                past_ep_rewards.append(e_reward)
                break
            else:
                s = s2
        if ep % update_iter == 0:
            states, actions, rewards, dones = history.get_history()
            gamma = .99
            discount = 0
            returns = [0] * len(rewards)

            for i in reversed(range(len(rewards))):
                if dones[i]:
                    returns[i] = rewards[i]
                else:
                    returns[i] = rewards[i] + discount
                discount = returns[i]*gamma
            returns = torch.FloatTensor(returns)
            mean = returns.mean()
            std = returns.std() + 1e-6
            returns = (returns - mean)/std

            logp = policy.get_log_p(states, actions)
            log_exp_return = torch.dot(returns, logp)/update_iter

            optimizer.zero_grad()

            (-log_exp_return).backward()
            optimizer.step()
            history.clear()

            update_viz(ep, sum(past_ep_rewards) / len(past_ep_rewards), algo_name)
            del past_ep_rewards[:]
train()
demo()
