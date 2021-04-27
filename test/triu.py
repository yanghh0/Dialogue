
import torch


x = torch.ones((1, 10, 10))
y = torch.triu(x, diagonal=1)
print(y)
print(1 - y)