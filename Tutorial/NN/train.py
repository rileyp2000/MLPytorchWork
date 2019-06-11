from network import Net
import numpy as np
import torch


net = Net()
#print(net)

params = list(net.parameters())
#print(len(params))
#print(params[0].size())

input = torch.randn(1,1,32,32)
print(input)
out = net(input)
print(out)
