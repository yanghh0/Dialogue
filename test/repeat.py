
import torch

x = torch.tensor([[1], [2], [3]])
print(x.shape)
xnew = x.repeat(2, 2, 3)
print(xnew.shape)
print(xnew)