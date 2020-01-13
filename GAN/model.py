import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, size):
        super(Generator, self).__init__()
        self.size = size

        self.main = nn.Sequential(
                                  nn.Linear(self.size, self.size),
                                  nn.ReLU(),
                                  nn.Linear(self.size, self.size),
                                  nn.ReLU(),
                                  nn.Linear(self.size, self.size),
                                  nn.ReLU(),
                                  nn.Linear(self.size, self.size),
                                  nn.Tanh(),
        )

    def forward(self, x):
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, size):
        super(Discriminator, self).__init__()
        self.size = size

        self.main = nn.Sequential(
                                  nn.Linear(self.size, self.size),
                                  nn.ReLU(),
                                  nn.Linear(self.size, self.size),
                                  nn.ReLU(),
                                  nn.Linear(self.size, self.size),
                                  nn.ReLU(),
                                  nn.Linear(self.size, 1),
                                  nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.main(x)
        return x