
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

    def scaled_dot_product(self, Q, K, V, mask=None):
        # QK^T/sqrt(dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        scores = scores.masked_fill(mask == 0, -1e9) if mask else scores
        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, V)
        return output, attn

    def forward(self, Q, K, V, mask=None):
        """shape of Q, K, V: (b, t, h)
        """
        batch_size = Q.size(0)

        Q = self.w_qs(Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.w_ks(K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.w_vs(V).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        X, attn = self.scaled_dot_product(Q, K, V, mask=mask)

        X = X.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.d_v)
        X = self.fc(X)

        return X, attn


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

