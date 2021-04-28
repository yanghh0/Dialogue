
import torch

a = torch.tensor([[[5, 6, 9, 3], 
                   [1, 2, 3, 6], 
                   [8, 3, 1, 3]], 

                  [[1, 4, 7, 0],
                   [2, 1, 9, 4],
                   [3, 8, 2, 2]]])

# 参数 mask 必须与 a 的 size 相同或者两者是可广播的
mask = torch.ByteTensor([[[1],
                          [1],
                          [0]],

                         [[0],
                          [1],
                          [1]]]).bool()

b = a.masked_fill(mask, value=torch.tensor(99))
print(b)