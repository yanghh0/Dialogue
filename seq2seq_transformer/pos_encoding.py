

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pos_table = torch.zeros(max_len, d_model).float()
        pos_table.require_grad = False

        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        pos_table[:, 0::2] = torch.sin(pos * div_term)
        pos_table[:, 1::2] = torch.cos(pos * div_term)
        pos_table = pos_table.unsqueeze(0)

        self.register_buffer('pos_table', pos_table)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


if __name__ == "__main__":
    # (b, t, h)
    inputs = torch.FloatTensor(5, 8, 10)
    use_gpu = True
    net = PositionalEncoding(10)
    if use_gpu:
        net.cuda()
        inputs = inputs.cuda()
    net(inputs)