
import sys
sys.path.append(".")
sys.path.append("..")


import torch
import torch.nn as nn
from seq2seq_transformer.attention import MultiHeadAttention


class SublayerConnection(nn.Module):
    """A residual connection followed by a layer norm.
       Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, n_head, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.attention = MultiHeadAttention(n_head, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        residual = q
        x, attn = self.attention(q, k, v, mask)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x, attn


if __name__ == "__main__":
    # (b, t, h)
    query = torch.randn(3, 8, 10)
    key = torch.randn(3, 5, 10)
    value = torch.randn(3, 5, 10)
    use_gpu = True
    net = SublayerConnection(n_head=1, d_model=10, dropout=0.1)
    if use_gpu:
        net.cuda()
        query = query.cuda()
        key = key.cuda()
        value = value.cuda()
    out = net(query, key, value)