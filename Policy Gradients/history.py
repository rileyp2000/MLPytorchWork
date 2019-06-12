import torch

class History():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.done = []

    def store(self, s, a, r, d):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.done.append(d)

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.done[:]

    def get_history(self):
        return self.states, self.actions, self.rewards, self.done
