

import torch
import torch.nn as nn
import torch.nn.functional as F


ground_truth = torch.LongTensor([[2, 0, 1],
                                 [2, 0, 1]])
label_mask = torch.sign(ground_truth).float()
print(label_mask)

pred = torch.Tensor([[[0.1, 0.2, 0.9],
                      [1.1, 0.1, 0.2],
                      [0.2, 2.1, 0.1]],

                     [[0.8, 0.2, 0.3],
                      [0.2, 0.3, 0.5],
                      [0.2, 0.2, 0.5]]])

rc_loss = F.cross_entropy(pred, ground_truth, reduce=False)
print(rc_loss)
rc_loss = torch.sum(rc_loss * label_mask, 1)
print(rc_loss)
print(rc_loss.mean())

print(F.cross_entropy(pred, ground_truth))
print(F.cross_entropy(pred, ground_truth, reduce=False))
print(F.cross_entropy(pred, ground_truth, reduce=False).mean())