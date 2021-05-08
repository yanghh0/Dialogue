
import torch

a = torch.Tensor([[1, 2, 4]])
b = torch.Tensor([[4, 5, 7], [3, 9, 8], [9, 6, 7]])
c = torch.cat((a, b), dim=0)
print(c)

d = torch.chunk(c, 2, dim=1)
print(d)