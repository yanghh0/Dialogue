
import sys
sys.path.append(".")
sys.path.append("..")


import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module
    """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid) # position-wise
        self.fc2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        # Add & Norm
        x += residual
        x = self.layer_norm(x)

        return x


if __name__ == "__main__":
    # (b, t, h)
    x = torch.randn(3, 8, 10)
    use_gpu = True
    net = PositionwiseFeedForward(d_in=10, d_hid=20)
    if use_gpu:
        net.cuda()
        x = x.cuda()
    out = net(x)