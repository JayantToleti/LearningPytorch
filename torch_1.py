#this script is for pytorch tutorial
import torch

import numpy as np
# 1) Tensors


# 1.1 Creating tensors

x = torch.empty(2,2)
print(x)

#tensor([[0., 0.],
        # [0., 0.]])
  
# You can fill it with zeroes (torch.zeros) or ones (torch.ones) or random values (torch.rand)

#  1.2 Operations

x = torch.rand(2,2)
y = torch.rand(2,2)
z= torch.add(x,y) # can also be torch.sub(x,y), torch.mul(x,y), torch.div(x,y)


# we can also do every thing in place with the underscore version(y.add_(x), y.sub_(x), etc)
print(z)

# we can also slice tensors using the colon operator like this: x[:,1]

x = torch.rand(5,3)
print(x)
print(x[:, 0]) # all rows in column 1
print(x[1, :]) # all columns in row 1
print(x[1, 1]) # value in row 1, column 1
print(x[1, 1].item()) # value in row 1, column 1

# reshaping a tensor

x = torch.rand(4,4)
print(x)
print(x.view(16)) # reshape to 16 elements
print(x.view(-1,8)) # reshape to 2 rows, 8 columns      
# 1.3 Numpy Bridge
a = torch.ones(5)
print(a)
b = a.numpy()
print(type(b))

# but changes in the tensor will be reflected in the numpy array

# so we have to be careful when we are working with numpy arrays and tensors we should first convert the numpy array to a tensor

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    x = torch.ones(5, device=device)  # directly create a tensor on GPU
    y = torch.ones(5)                       # or just use strings ``.to("cuda")``
    y = y.to(device)
    z = x + y
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!