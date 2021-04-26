
import sys
sys.path.append(".")
sys.path.append("..")


import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    def __init__(self, method, hidden_size):
        super(LuongAttention, self).__init__()
        if method not in ['dot', 'general', 'concat']:
            raise ValueError(method, "is not an appropriate attention method.")

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_outputs):
        # shape of hidden: (t=1, b, h)
        # shape of encoder_outputs: (t, b, h)
        return torch.sum(hidden * encoder_outputs, dim=2)

    def general_score(self, hidden, encoder_outputs):
        energy = self.attn(encoder_outputs)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_outputs):
        energy = self.attn(torch.cat((hidden.expand(encoder_outputs.size(0), -1, -1), encoder_outputs), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose seq_length and batch_size dimensions
        # shape of attn_energies: (b, t)
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        # return shape: (b, 1, t)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


if __name__ == "__main__":
    # (t=1, b, h)
    hidden = torch.randn(1, 3, 10)
    encoder_outputs = torch.randn(5, 3, 10)

    use_gpu = True
    net = LuongAttention('dot', 10)
    if use_gpu:
        net.cuda()
        hidden = hidden.cuda()
        encoder_outputs = encoder_outputs.cuda()
    out = net(hidden, encoder_outputs)
    print(out)

