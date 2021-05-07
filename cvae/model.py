
import sys
sys.path.append(".")
sys.path.append("..")


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tensorflow as tf
from cvae.config import Config


class KgRnnCVAE(nn.Module):
    def __init__(self, word_vocab_size, topic_vocab_size, act_vocab_size):
        super(KgRnnCVAE, self).__init__()
        self.word_vocab_size = word_vocab_size
        self.topic_vocab_size = topic_vocab_size
        self.act_vocab_size = act_vocab_size

        self.word_embedding = nn.Embedding(self.word_vocab_size, Config.word_embed_size, padding_idx=0)
        self.topic_embedding = nn.Embedding(self.topic_vocab_size, Config.topic_embed_size)
        if self.use_hcf:
            self.act_embedding = nn.Embedding(self.act_vocab_size, Config.act_embed_size)


if __name__ == "__main__":
    pass


