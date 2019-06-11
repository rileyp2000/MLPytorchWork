from __future__ import print_function
import torch
import numpy as np

"""
#--------------
Tensor Practice
#--------------

x = torch.tensor([5.5, 3])
x = x.new_ones(5,3, dtype=torch.double)
#print(x)
x = torch.randn_like(x, dtype=torch.float)
#print(x)

#print(x.shape)
y = x.view(15)
z = x.view(-1,5)
#print(x.shape, y.shape, z.shape)

a = torch.ones(5)
#print(a)

b = a.numpy()
#print(b)
a.add_(1)
#print(b)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
#print(a)
#print(b)

#----- Autograd and backprop -----#

x = torch.ones(2,2,requires_grad=True)
#print(x)

y = x + 2
#print(y)
z = y * y * 3
out = z.mean()
#print(z,out)

#Gradients stuff
out.backward()
print(x.grad)


x = torch.randn(3, requires_grad=True)

y = x*2
#print(y)
while y.data.norm() < 1000:
    y *= 2
#print(y)


print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)"""
