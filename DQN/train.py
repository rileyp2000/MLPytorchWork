import gym
import numpy as np
import torch
import random

from model import *
from replay_buffer import *
#from visualize import *


env = gym.make('CartPole-v1')
epsilon = .2
q = Q(env)
optimizer = torch.optim.Adam(q.parameters(), lr=1e-3)

batch_size = 128
rb = ReplayBuffer(1e5)

def train():
    s = env.reset()
    while True:
        if random.random() < epsilon:
            a = env.action_space.sample()
        else:
            max_a = np.zeros(1)
            for a in range(1, env.action_space.n):
                if q(s,a) > q(s, max_a):
                    max_a = a
            a = int(max_a)

        s2, r, done, info = env.step(int(a))
        rb.store(s, a, r, s2, done)

        if done:
            break
        else:
            s = s2

        update()



def update():
    s, a, r, s2, m = rb.sample(batch_size)
    print(s.shape, a.shape, r.shape, s2.shape, m.shape)


train()
