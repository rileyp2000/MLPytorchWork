import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class PolicyGradient(nn.Module):
    def __init__(self,env):
        super(PolicyGradient, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ELU(),
            nn.Linear(64,64),
            nn.ELU(),
            nn.Linear(64, env.action_space.shape[0]),
            #nn.Tanh()
        )

    def forward(self, s):
        # a = self.main(torch.FloatTensor(s))
        # return map(a, env.action_space.n)
        #quit(self.main(torch.FloatStorage(s)))
        return self.main(torch.FloatTensor(s))

class Q(nn.Module):
    def __init__(self,env, isMulti):
        super(Q, self).__init__()
        self.multi = isMulti
        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0] + env.action_space.shape[0],64),
            nn.ELU(),
            nn.Linear(64,64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, s, a):
        if self.multi:
            return self.main(torch.cat([s,a],2))
        else:
            return self.main(torch.cat([s,a],1))
