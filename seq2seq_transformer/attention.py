
import sys
sys.path.append(".")
sys.path.append("..")


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    """
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()

        assert d_model % n_head == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // n_head
        self.d_v = self.d_k
        self.n_head = n_head

        self.w_qs = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.w_ks = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.w_vs = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.fc = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def scaled_dot_product(self, q, k, v, mask=None):
        # QK^T/sqrt(dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(mask == 0, -1e9) if mask else scores
        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

    def forward(self, q, k, v, mask=None):
        """shape of q, k, v: (b, t, h)
        """
        batch_size = q.size(0)

        q = self.w_qs(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_ks(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(v).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        x, attn = self.scaled_dot_product(q, k, v, mask=mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.d_v)
        x = self.fc(x)

        return x, attn


if __name__ == "__main__":
    # (b, t, h)
    query = torch.randn(3, 8, 10)
    key = torch.randn(3, 5, 10)
    value = torch.randn(3, 5, 10)
    use_gpu = True
    net = MultiHeadAttention(n_head=1, d_model=10)
    if use_gpu:
        net.cuda()
        query = query.cuda()
        key = key.cuda()
        value = value.cuda()
    out = net(query, key, value)

