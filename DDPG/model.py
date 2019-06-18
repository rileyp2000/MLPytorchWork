import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class Actor(nn.Module):
    def __init__(self,env):
        super(Actor, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0],64),
            nn.ELU(),
            nn.Linear(64,64),
            nn.ELU(),
            nn.Linear(64, env.action_space.n),
            nn.Tanh()
        )

    def forward(self, s):
        a = self.main(torch.FloatTensor(s))
        return self.main(torch.FloatTensor(s))


class Critic(nn.Module):
    def __init__(self,env):
        super(Critic, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0],64),
            nn.ELU(),
            nn.Linear(64,64),
            nn.ELU(),
            nn.Linear(64, env.action_space.n),
            nn.Tanh()
        )

    def forward(self, s):
        return self.main(torch.FloatTensor(s))
