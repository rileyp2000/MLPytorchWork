import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import A2C
from history import History

num_episodes = 1000

gamma = .99
learn_rate = 2e-3

env = gym.make('CartPole-v1')
history = History()

a2c = A2C(env)
optimizer = torch.optim.Adam(a2c.parameters(), lr=learn_rate)

def train():
    for ep in range(num_episodes):
        s = env.reset()

        while True:
            s = torch.FloatTensor(s).unsqueeze(0)

            action_probs = a2c.get_action_probs(s)
            quit(action_probs)
            a = action_probs.multinomial().data[0][0]

            #quit(s.dtype)








train()
