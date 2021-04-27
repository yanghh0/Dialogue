
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


if __name__ == "__main__":
    src_seqs = torch.LongTensor([[1, 2, 3, 4],
                                 [4, 3, 2, 0],
                                 [7, 4, 0, 0]])
    trg_seqs = torch.LongTensor([[3, 1, 3, 4],
                                 [4, 6, 2, 0],
                                 [7, 7, 0, 0]])

    pad_mask = get_pad_mask(src_seqs, pad_idx=0)
    subsequent_mask = get_subsequent_mask(trg_seqs)

    print(pad_mask)
    print(subsequent_mask)