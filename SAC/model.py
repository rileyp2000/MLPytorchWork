import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Policy(nn.Module):
    def __init__(self, env):
        super(Policy, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(64, env.action_space.shape[0])
        )
        self.sigma = nn.Sequential(
            nn.Linear(64, env.action_space.shape[0])
        )

    def dist(self, s):
        main = self.actor(torch.FloatTensor(s))
        mu = self.mu(main)
        sigma = torch.exp(self.sigma(main))
        dist = Normal(mu, sigma)
        return dist

    def forward(self, s):
        dist = self.dist(s)
        return torch.tanh(dist.sample())

    def sample(self, s):
        dist = self.dist(s)
        x = dist.rsample()
        a = torch.tanh(x)

        log_p = dist.log_prob(x)
        log_p -= torch.log(1 - torch.pow(a,2) + 1e-6)

        return a, log_p


class Q(nn.Module):
    def __init__(self,env):
        super(Q, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0] + env.action_space.shape[0],64),
            nn.ELU(),
            nn.Linear(64,64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, s, a):
        return self.main(torch.cat([s,a],1))
