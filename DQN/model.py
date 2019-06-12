import torch
import torch.nn as nn


class Q(nn.Module):
    def __init__(self,env):
        super(Q, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0] + 1, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    def forward(self, s, a):
        return self.main(torch.cat([torch.FloatTensor(s),torch.FloatTensor(a)], -1))
