
import torch


T1 = torch.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
T2 = torch.tensor([[10, 20, 30],
                   [40, 50, 60],
                   [70, 80, 90]])

print(torch.stack((T1, T2)))
print(torch.stack((T1, T2)).shape)
print(torch.stack((T1, T2), dim=1))
print(torch.stack((T1, T2), dim=1).shape)