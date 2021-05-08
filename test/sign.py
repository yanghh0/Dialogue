
import torch

a = torch.randn(4)
print(a)
# 符号函数，返回一个新张量，包含输入input张量每个元素的正负（大于0的元素对应1，小于0的元素对应-1，0还是0）
print(torch.sign(a))