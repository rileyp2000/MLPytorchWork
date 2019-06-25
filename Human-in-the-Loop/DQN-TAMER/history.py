import torch

class History():
    def __init__(self):
        self.states = []
        self.actions = []
        self.feedback = []

    def store(self, s, a, f):
        self.states.append(s)
        self.actions.append(a)
        self.feedback.append(f)

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.feedback[:]

    def get_history(self):
        return self.states, self.actions, self.feedback
