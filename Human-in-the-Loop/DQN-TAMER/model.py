import torch
import torch.nn as nn


class Q(nn.Module):
    def __init__(self,env):
        super(Q, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, env.action_space.n)
        )

    def forward(self, s):
        return self.main(torch.FloatTensor(s))

class H(nn.Module):
    def __init__(self, env):
        super(H, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0],64),
            nn.ELU(),
            nn.Linear(64,16),
            nn.ELU(),
            nn.Linear(16,16),
            nn.ELU(),
            nn.Linear(16, env.action_space.n)

        )

    def forward(self,s):
        return self.main(torch.FloatTensor(s))
        #TODO See if the network needs to actually return the binary feedback for the state versus the probability of actions
