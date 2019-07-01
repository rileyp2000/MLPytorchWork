import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import *
from replay_buffer import ReplayBuffer
from visualize import *
from copy import deepcopy
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

def make_env():
    def _f():
        return gym.make('Pendulum-v0')
    return _f

num_actors = 4
env = SubprocVecEnv([make_env() for _ in range(num_actors)])

algo_name = 'DDPG Multi-Agent'
max_ts = 100000

gamma = .99
learn_rate = 3e-4
tau = .995

rb = ReplayBuffer(1e6, True)
batch_size = 128

policy = PolicyGradient(env)
policy_target = deepcopy(policy)
pol_optim = torch.optim.Adam(policy.parameters(), learn_rate)

q = Q(env, True)
q_target = deepcopy(q)
q_optim = torch.optim.Adam(q.parameters(),lr=learn_rate)

def train():
    s = env.reset()
    explore(10000)
    ep_r = np.zeros(num_actors)
    ep_r_final = np.zeros(num_actors)
    ts = 0
    while ts < max_ts:
        with torch.no_grad():
            a = policy(s) + addNoise()
        s2, r, done, _ = env.step(a)
        #print(s.shape, s2.shape, a.shape, r.shape, done.shape)
        rb.store(s,a,r,s2,done)
        ep_r += r
        ts += 1

        mask = 1 - done
        ep_r_final = (ep_r_final * mask) + (done * ep_r)
        ep_r *= mask
        if ts % 200 == 0:
            update_viz(ts, ep_r_final, algo_name)

        s = s2
        update()

#Explores the environment for the specified number of timesteps to improve the performance of the DQN
def explore(timestep):
    #print('Exploring...')
    for ts in range(timestep):
        s = env.reset()
        a = np.stack([env.action_space.sample() for i in range(num_actors)])
        s2, r, done, _ = env.step(a)
        rb.store(s, a, r, s2, done)
        s = s2

def addNoise():
    return np.clip(np.random.normal(0,.15), -.3,.3)

def update():
    s, a, r, s2, m = rb.sample(batch_size)
    #print(s.shape, s2.shape, a.shape, r.shape, m.shape)
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
