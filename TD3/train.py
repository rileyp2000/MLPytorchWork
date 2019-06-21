import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import *
from replay_buffer import ReplayBuffer
from visualize import *
from copy import deepcopy


algo_name = 'TD3'
max_episode = 2000
policy_delay = 2

gamma = .9
learn_rate = 3e-4
tau = .995

env = gym.make('Pendulum-v0')
rb = ReplayBuffer(1e6)
batch_size = 128

policy = PolicyGradient(env)
policy_target = deepcopy(policy)
pol_optim = torch.optim.Adam(policy.parameters(), learn_rate)

q1 = Q(env)
q2 = deepcopy(q1)
q1_target = deepcopy(q1)
q2_target = deepcopy(q2)
q1_optim = torch.optim.Adam(q1.parameters(),learn_rate)
q2_optim = torch.optim.Adam(q2.parameters(), learn_rate)

#Trains the network according to the TD3 algorithm
def train():
    explore(100000)
    ep = 0
    up_ct = 0
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
            update(up_ct)
            up_ct += 1

#Updates all the networks involved at their proper intervals
def update(up_ct):
    s, a, r, s2, m = rb.sample(batch_size)
    a = a.squeeze().unsqueeze(1)

    #Finds the max next action according to the policy then finds the min between the two q target networks
    with torch.no_grad():
        max_next_a = policy_target(s2) + np.clip(addNoise(), -.4,.4)
        q1_next_max = q1_target(s2, max_next_a)
        q2_next_max = q2_target(s2, max_next_a)
        min_q = torch.min(q1_next_max, q2_next_max)

        #target for both loss functions
        y = r + m*gamma*min_q

    q1_loss = F.mse_loss(q1(s, a), y)
    q2_loss = F.mse_loss(q2(s, a), y)

    #Update both q networks every update step
    q1_optim.zero_grad()
    q1_loss.backward()
    q1_optim.step()

    q2_optim.zero_grad()
    q2_loss.backward()
    q2_optim.step()

    policyUpdate(s, a, r, s2, m, up_ct)



#Updates the policy, but only on the correct interval
def policyUpdate(s, a, r, s2, m, up_ct):
    if up_ct % policy_delay == 0:
        policy_loss = -(q1(s, policy(s))).mean()

        pol_optim.zero_grad()
        policy_loss.backward()
        pol_optim.step()

        #Update q_target and policy_target
        for param, target_param in zip(q1.parameters(), q1_target.parameters()):
            target_param.data = target_param.data*tau + param.data*(1-tau)
        for param, target_param in zip(q2.parameters(), q2_target.parameters()):
            target_param.data = target_param.data*tau + param.data*(1-tau)

        for param, target_param in zip(policy.parameters(), policy_target.parameters()):
            target_param.data = target_param.data*tau + param.data*(1-tau)
    else:
        return

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

#Adds random noise to the action choice to encourage exploration
def addNoise():
    return np.random.normal(0,.15)




train()
