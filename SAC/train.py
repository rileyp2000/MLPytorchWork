import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import *
from replay_buffer import ReplayBuffer
from visualize import *
from copy import deepcopy


algo_name = 'SAC'
max_episode = 2000

gamma = .99
alpha = .95
learn_rate = 1e-4
tau = .995

env = gym.make('Pendulum-v0')
rb = ReplayBuffer(1e6)
batch_size = 128

policy = Policy(env)
pol_optim = torch.optim.Adam(policy.parameters(), learn_rate)

q1 = Q(env)
q2 = deepcopy(q1)
q1_target = deepcopy(q1)
q2_target = deepcopy(q2)
q1_optim = torch.optim.Adam(q1.parameters(),learn_rate)
q2_optim = torch.optim.Adam(q2.parameters(), learn_rate)


def train():
    explore(10000)
    ep = 0
    while ep < max_episode:
        s = env.reset()
        ep_r = 0
        while True:
            with torch.no_grad():
                a = policy(s)
            s2, r, done, _ = env.step(2*a)
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


def update():
    s, a, r, s2, m = rb.sample(batch_size)
    a = a.squeeze().unsqueeze(1)
    with torch.no_grad():
        a2, p2 = policy.sample(s2)
        q1_next_max = q1_target(s2, a2)
        q2_next_max = q2_target(s2, a2)
        min_q = torch.min(q1_next_max, q2_next_max)

        y = r + m*gamma*(min_q - alpha*p2)

    q1_loss = F.mse_loss(q1(s, a), y)
    q2_loss = F.mse_loss(q2(s, a), y)

    #Update q and policy with backprop
    q1_optim.zero_grad()
    q1_loss.backward()
    q1_optim.step()

    q2_optim.zero_grad()
    q2_loss.backward()
    q2_optim.step()


    new_a, p = policy.sample(s)

    policy_loss = (alpha*p - q1(s, new_a)).mean()

    pol_optim.zero_grad()
    policy_loss.backward()
    pol_optim.step()

    #Update q_target and policy_target
    for param, target_param in zip(q1.parameters(), q1_target.parameters()):
        target_param.data = target_param.data*tau + param.data*(1-tau)
    for param, target_param in zip(q2.parameters(), q2_target.parameters()):
        target_param.data = target_param.data*tau + param.data*(1-tau)


train()
