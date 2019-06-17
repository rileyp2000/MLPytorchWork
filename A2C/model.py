import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import copy

class Actor(nn.Module):
    def __init__(self,env):
        super(Actor, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0],64),
            nn.ELU(),
            nn.Linear(64,64),
            nn.ELU(),
            nn.Linear(64, env.action_space.n)
        )

    def forward(self, s):
        logits = self.main(torch.FloatTensor(s))
        dist = Categorical(logits=logits)
        a = dist.sample().numpy()
        return a

    def get_log_p(self, s, a):
        logits = self.main(torch.FloatTensor(s))
        dist = Categorical(logits=logits)

        log_p = dist.log_prob(torch.FloatTensor(np.array(a)))

        return log_p


class Critic(nn.Module):
    def __init__(self,env):
        super(Critic, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0],64),
            nn.ELU(),
            nn.Linear(64,64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    def forward(self, s):
        return self.main(torch.FloatTensor(s))


class A2C(nn.Module):
    def __init__(self,env):
        super(A2C, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0],64),
            nn.ELU(),
            nn.Linear(64,128),
            nn.ELU(),
            nn.Linear(128,64),
            nn.ELU()
        )

        self.actor = nn.Sequential(
            nn.Linear(64, env.action_space.n)
        )

        self.critic = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, s):
        return self.main(torch.FloatTensor(s))

    def get_action_probs(self, s):
        s = self(s)
        return F.softmax(self.actor(s))

    def get_state_value(self, s):
        s = self(s)
        return self.critic(s)

    def evaluate_actions(self, s):
        action_probs = self.get_action_probs(s)
        state_value = self.get_state_value(s)
        return action_probs, state_value
