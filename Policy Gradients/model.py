import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(4, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 2)
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
