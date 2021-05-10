

import sys
sys.path.append(".")
sys.path.append("..")


import torch
import torch.nn as nn
from seq2seq_cvae.config import Config


class ActFC(nn.Module):
    def __init__(self):
        super(ActFC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(Config.act_embed_size, 30), 
            nn.Tanh()
        )

    def forward(self, inputs):
        """shape of inputs: (60, 30)
        """
        return self.fc(inputs)


if __name__ == "__main__":
    model = ActFC()
    model.cuda()