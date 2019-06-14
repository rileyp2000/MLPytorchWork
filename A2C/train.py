import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from history import History

max_steps = 10000
num_steps = 128
steps = 0

gamma = .99
epsilon = .1
learn_rate = 2e-3

env = gym.make('CartPole-v1')
history = History()

actor = Actor(env)
opt_A = torch.optim.Adam(actor.parameters(), lr=learn_rate)

critic = Critic(env)
opt_C = torch.optim.Adam(critic.parameters(), lr=learn_rate)


def train():
    global steps
    s = env.reset()
    while steps < max_steps:
        n_s = 0
        #Collecting trajectories
        while n_s < num_steps:
            a = actor(s)
            s2, r, done, _ = env.step(a)
            history.store(s, a, r, done)
            if done:
                s = env.reset()
            else:
                s = s2
            n_s += 1
            steps += 1
        n_s = 0
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
        advantage = returns - critic(s)

        logp = actor.get_log_p(states, actions)
        policy_loss = -(logp * advantage).mean()

        opt_A.zero_grad()
        policy_loss.backward()
        opt_A.step()

        #------Value Function loss and update------#
        v = F.mse_loss(returns, critic(states).squeeze(1))

        opt_C.zero_grad()
        v.backward()
        opt_C.step()

        history.clear()

train()
