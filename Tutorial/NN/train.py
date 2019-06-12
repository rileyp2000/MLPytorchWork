from network import Net
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from visualize import *


net = Net()
#print(net)


params = list(net.parameters())
#print(len(params))
#print(params[0].size())
#Update weights
optimizer = optim.Adam(net.parameters(), lr=.01)

input = torch.randn(1,1,32,32)
#print(input)
out = net(input)
#print(out)

#Zeros gradient
net.zero_grad()

target = torch.randn(10)
#reshape
target = target.view(1,-1)
#print("Target: {0}".format(target))
criterion = nn.MSELoss()
#print(type(criterion))
def train():
    for i in range(0,100):
        optimizer.zero_grad()
        output = net(input)
        print(output)
        print(target)
        loss = criterion(output, target)
        print('Loss: {0}'.format(loss))
        loss.backward()
        optimizer.step()
        update_viz(i, loss.item(), "Pytorch Ex Loss")

train()
