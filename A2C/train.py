import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from history import History
from visualize import *

algo_name = 'A2C'

max_steps = 100000
num_steps = 128

gamma = .9
learn_rate = 1e-3

env = gym.make('CartPole-v1')
history = History()

actor = Actor(env)
opt_A = torch.optim.Adam(actor.parameters(), lr=learn_rate)

critic = Critic(env)
opt_C = torch.optim.Adam(critic.parameters(), lr=learn_rate)


def train():
    steps = 0
    s = env.reset()
    ep = 0
    while steps < max_steps:
        n_s = 0
        ep_r = 0
        #Collecting trajectories
        while n_s < num_steps:
            a = actor(s)
            s2, r, done, _ = env.step(a)
            history.store(s, a, r, done)
            ep_r += r
            if done:
                update_viz(ep, ep_r, algo_name)
                ep += 1
                s = env.reset()
            else:
                s = s2
            n_s += 1
            steps += 1
        #----------#
        #  Update  #
        #----------#
        states, actions, rewards, dones = history.get_history()

        #Calculate the returns and normalize them
        discount = 0
        returns = [0] * len(rewards)
        for i in reversed(range(len(rewards))):
            if dones[i]:
                returns[i] = rewards[i]
            else:
                returns[i] = rewards[i] + discount
            discount = returns[i]*gamma
        returns = torch.FloatTensor(returns)

        #------Policy Loss and Update------#
        #Get advantage

        advantage = returns.unsqueeze(1) - critic(states)
        mean = advantage.mean()
        std = advantage.std() + 1e-6
        advantage = (advantage - mean)/std


        logp = actor.get_log_p(states, actions)

        policy_loss = (-(logp.unsqueeze(1) * advantage)).mean()

        opt_A.zero_grad()
        policy_loss.backward()
        opt_A.step()

        #------Value Function loss and update------#
        # print(returns.shape)
        # quit(critic(states).shape)
        v = F.mse_loss(returns.unsqueeze_(1), critic(states))

        opt_C.zero_grad()
        v.backward()
        opt_C.step()

        history.clear()

train()
