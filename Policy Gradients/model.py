import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class Policy(nn.Module):
    def __init__(self, env):
        super(Policy, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, env.action_space.n)
        )

    def forward(self, s):
        logits = self.actor(torch.FloatTensor(s))
        dist = Categorical(logits=logits)
        a = dist.sample().numpy()

        return a

    def get_log_p(self, s, a):
        logits = self.actor(torch.FloatTensor(s))
        dist = Categorical(logits=logits)

        log_p = dist.log_prob(torch.FloatTensor(np.array(a)))

        return log_p
