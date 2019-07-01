import torch

class History():
    def __init__(self, size):
        self.states = []
        self.actions = []
        self.feedback = []
        self.max_size = size

    def store(self, s, a, f):
        if self.max_size == len(self.states):
            self.clear()
        self.states.append(s)
        self.actions.append(a)
        self.feedback.append(f)

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.feedback[:]

    def get_history(self):
        return self.states, self.actions, self.feedback
#TODO add a way to change the feedback values for the storage that will work with global buffer
