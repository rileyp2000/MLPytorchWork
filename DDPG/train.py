import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import *
from replay_buffer import ReplayBuffer
from visualize import *
from copy import deepcopy


algo_name = 'DDPG'
max_episode = 2000

gamma = .9
learn_rate = 1e-4
tau = .995

env = gym.make('Pendulum-v0')
rb = ReplayBuffer(1e6)
batch_size = 128

policy = PolicyGradient(env)
policy_target = deepcopy(policy)
pol_optim = torch.optim.Adam(policy.parameters(), learn_rate)

q = Q(env)
q_target = deepcopy(q)
q_optim = torch.optim.Adam(q.parameters(),lr=learn_rate)

def train():
    explore(10000)
    ep = 0
    while ep < max_episode:
        s = env.reset()
        ep_r = 0
        while True:
            with torch.no_grad():
                a = policy(s) + addNoise()
            s2, r, done, _ = env.step(a)
            rb.store(s,a,r,s2,done)
            ep_r += r

            if done:
                update_viz(ep, ep_r, algo_name)
                ep +=1
                break
            else:
                s = s2
            update()

#Explores the environment for the specified number of timesteps to improve the performance of the DQN
def explore(timestep):
    ts = 0
    while ts < timestep:
        s = env.reset()
        while True:
            a = env.action_space.sample()
            s2, r, done, _ = env.step(a)
            rb.store(s, a, r, s2, done)
            ts += 1
            if done:
                break
            else:
                s = s2

def addNoise():
    return np.random.normal(0,.15)

def update():
    s, a, r, s2, m = rb.sample(batch_size)
    #print(s.shape, a.shape, r.shape, s2.shape, m.shape)
    a = a.squeeze().unsqueeze(1)
    #print(s.shape, a.shape, r.shape, s2.shape, m.shape)
    # quit()
    with torch.no_grad():
        max_next_a = policy_target(s2)
        y = r + m*gamma*q_target(s2, max_next_a)
    q_loss = F.mse_loss(q(s, a), y)


    #Update q and policy with backprop
    q_optim.zero_grad()
    q_loss.backward()
    q_optim.step()

    policy_loss = -(q(s, policy(s))).mean()
    pol_optim.zero_grad()
    policy_loss.backward()
    pol_optim.step()

    #Update q_target and policy_target
    for param, target_param in zip(q.parameters(), q_target.parameters()):
        target_param.data = target_param.data*tau + param.data*(1-tau)
    for param, target_param in zip(policy.parameters(), policy_target.parameters()):
        target_param.data = target_param.data*tau + param.data*(1-tau)


train()
