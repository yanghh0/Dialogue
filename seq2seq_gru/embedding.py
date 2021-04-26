
import sys
sys.path.append(".")
sys.path.append("..")


import torch
import torch.nn as nn
import numpy as np


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(WordEmbedding, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self._init_embedding()

    def _init_embedding(self):
        pretrain_emb = np.empty([self.vocab_size, self.embedding_dim])
        scale = np.sqrt(3.0 / self.embedding_dim)
        for index in range(self.vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])
        self.embedding.weight.data.copy_(torch.from_numpy(pretrain_emb))

    def forward(self, inputs):
        """shape of inputs: (t, b)
        """
        embeded = self.embedding(inputs)
        return embeded


if __name__ == "__main__":
    inputs = torch.LongTensor([[1, 4, 7],
                               [2, 3, 4],
                               [3, 2, 0],
                               [4, 0, 0]])
    use_gpu = True
    net = WordEmbedding(vocab_size=10, embedding_dim=6)
    if use_gpu:
        net.cuda()
        inputs = inputs.cuda()
    output = net(inputs)
    print(output)