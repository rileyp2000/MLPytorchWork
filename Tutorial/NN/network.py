import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #1 input image channel, 6 outpuct channels 3x3 square convolution (movey boxes)
        """self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)"""

        self.conv = nn.Sequential(
            nn.Conv2d(1,6,3),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(6,16,3),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )
        self.lin = nn.Sequential(
            nn.Linear(1024, 120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

    def forward(self, x):
        """x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)"""

        logits = self.conv(torch.FloatTensor(x))
        x = x.view(-1, self.num_flat_features(x))
        #print(x.size())

        return self.lin(x)

        #dist = Categorical(logits=logits)
        #x = dist.sample().numpy()
        #return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
