
import sys
sys.path.append(".")
sys.path.append("..")


import torch


def get_pad_mask(inputs, pad_idx):
    """shape of inputs: (b, t)
       return shape: (b, 1, t)
    """
    pad_mask = (inputs != pad_idx).unsqueeze(-2)
    return pad_mask


def get_subsequent_mask(targets):
    """shape of targets: (b, t')
       return shape: (1, t', t')
    """
    b, t = targets.size()
    subsequent_mask = (1 - torch.triu(torch.ones((1, t, t), device=targets.device), diagonal=1)).bool()
    return subsequent_mask