import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from model import *
from replay_buffer import *
from visualize import *

algo_name = 'DQN'
env = gym.make('CartPole-v1')
epsilon = .2
gamma = .99
q = Q(env)
optimizer = torch.optim.Adam(q.parameters(), lr=1e-3)
max_ep = 1000

batch_size = 128
rb = ReplayBuffer(1e5)


def train():
    ep = 0
    while ep < max_ep:
        s = env.reset()
        ep_r = 0
        while True:
            with torch.no_grad():
                if random.random() < epsilon:
                    a = env.action_space.sample()
                else:
                    a = int(np.argmax(q(s)))

            s2, r, done, info = env.step(int(a))
            rb.store(s, a, r, s2, done)
            ep_r += r

            if done:
                update_viz(ep, ep_r, algo_name)
                ep += 1
                break
            else:
                s = s2

            update()



def update():
    s, a, r, s2, m = rb.sample(batch_size)
    #print(s.shape, a.shape, r.shape, s2.shape, m.shape)
    max_next_q, _ = q(s2).max(dim=1, keepdim=True)
    #print(max_next_q.shape)
    y = r + m*gamma*max_next_q
    loss = F.mse_loss(torch.gather(q(s), 1, a.long()), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




train()
